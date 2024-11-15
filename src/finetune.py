import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import (
    init_process_group,
    is_initialized
)
from src.models.attentive_pooler import AttentiveClassifier
from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.imagenet1k import make_imagenet1k
from src.datasets.custom_image_dataset import make_adni, make_dzne, make_hospital

from src.helper import (
    load_checkpoint,
    init_model,
    init_finetune_model,
    init_opt,
    load_checkpoint_ft,
    init_opt_ft
    )
from src.transforms import make_transforms, make_transforms_adni
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

import wandb

# --
log_timings = True
log_freq = 10

# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    proj_name = args['meta']['proj_name']
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    resume_checkpoint = args['meta']['resume_checkpoint']
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    finetune = args['meta']['finetune']
    pretrain_ds = args['meta']['pretrain_ds']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    # --
    dataset = args['data']['dataset']
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    in_chans = args['data']['in_chans']
    num_classes = args['data']['num_classes']
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    disable_wandb = args['logging']['disable_wandb']

    checkpoint_freq = num_epochs #only save the last model

    if not os.path.exists(folder):
        os.makedirs(folder)
    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()

    if not is_initialized():
        init_process_group(
            backend='nccl',  # or 'gloo' depending on your setup
            init_method='tcp://localhost:13147',  # environment variable-based initialization
            world_size=world_size,
            rank=rank
        )

    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}-ft-r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}-ft-{dataset}' + '-ep{epoch}.pth.tar')
    best_path = os.path.join(folder, f'{tag}-ft-{dataset}-best.pth.tar')
    load_path = None
    if load_model:
        load_path = r_file

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                          ('%d', 'time (ms)'))

    if rank == 0 and not disable_wandb:
        wandb_id = wandb.util.generate_id()

    if not disable_wandb:
        run = wandb.init(project=f"{proj_name}_ft_{dataset}", 
                        name=f"{model_name}_{pretrain_ds}_atp_nfe_bs{batch_size}_ep{num_epochs}_lr{lr}_5folds", 
                        config=args,
                        id=wandb_id,
                        resume='allow',
                        dir=folder)
                # define your custom x axis metric
        wandb.define_metric("custom_step")

    # 5 folds 
    fivefolds_test_loss = []
    fivefolds_test_accuracy = []
    fivefolds_test_bal_accuracy = []
    fivefold_test_precision = []
    fivefold_test_recall = []
    fivefold_test_f1 = []

    for f in range(1,6):
        if not disable_wandb:
            # define which metrics will be plotted against it
            wandb.define_metric(f"Fold {f} - lr", step_metric="custom_step")
            wandb.define_metric(f"Fold {f} - Training Loss", step_metric="custom_step")
            wandb.define_metric(f"Fold {f} - Validation Loss", step_metric="custom_step")

        #when do we use the folds? dataloader?
        print(f"################ Fold {f} ####################")
        # -- init model
        # for reading from the checkpoint
        encoder, predictor = init_model(
            device=device,
            patch_size=patch_size,
            crop_size=crop_size,
            pred_depth=pred_depth,
            pred_emb_dim=pred_emb_dim,
            in_chans=in_chans,
            model_name=model_name)
        target_encoder = copy.deepcopy(encoder)


        if dataset in ['dzne', 'hospital']:
            ##TODO: for finetuning we don't need masks anymore right?
            # mask_collator = MBMaskCollator(
            #     input_size=crop_size,
            #     patch_size=patch_size,
            #     pred_mask_scale=pred_mask_scale,
            #     enc_mask_scale=enc_mask_scale,
            #     aspect_ratio=aspect_ratio,
            #     nenc=num_enc_masks,
            #     npred=num_pred_masks,
            #     allow_overlap=allow_overlap,
            #     min_keep=min_keep)

            #adni transform
            transform = make_transforms_adni()

            if dataset == 'dzne':
                #create dzne dataloader
                _, train_loader, train_sampler = make_dzne(
                    transform=transform,
                    batch_size=batch_size,
                    # collator=mask_collator,
                    pin_mem=pin_mem,
                    training=True,
                    num_workers=num_workers,
                    world_size=world_size,
                    rank=rank,
                    root_path=root_path,
                    image_folder=image_folder,
                    copy_data=copy_data,
                    drop_last=True,
                    mode='train',
                    fold=f)
                
                _, validation_loader, validation_sampler = make_dzne(
                    transform=transform,
                    batch_size=batch_size,
                    # collator=mask_collator,
                    pin_mem=pin_mem,
                    training=False,
                    num_workers=num_workers,
                    world_size=world_size,
                    rank=rank,
                    root_path=root_path,
                    image_folder=image_folder,
                    copy_data=copy_data,
                    drop_last=True,
                    mode='val',
                    fold=f)
                
                _, test_loader, test_sampler = make_dzne(
                    transform=transform,
                    batch_size=batch_size,
                    # collator=mask_collator,
                    pin_mem=pin_mem,
                    training=False,
                    num_workers=num_workers,
                    world_size=world_size,
                    rank=rank,
                    root_path=root_path,
                    image_folder=image_folder,
                    copy_data=copy_data,
                    drop_last=True,
                    mode='test',
                    fold=f)
            else: 
                #create hospital dataloader
                _, train_loader, train_sampler = make_hospital(
                    transform=transform,
                    batch_size=batch_size,
                    # collator=mask_collator,
                    pin_mem=pin_mem,
                    training=True,
                    num_workers=num_workers,
                    world_size=world_size,
                    rank=rank,
                    root_path=root_path,
                    image_folder=image_folder,
                    copy_data=copy_data,
                    drop_last=True,
                    mode="train",
                    fold=f)
                
                _, validation_loader, validation_sampler = make_hospital(
                    transform=transform,
                    batch_size=batch_size,
                    # collator=mask_collator,
                    pin_mem=pin_mem,
                    training=False,
                    num_workers=num_workers,
                    world_size=world_size,
                    rank=rank,
                    root_path=root_path,
                    image_folder=image_folder,
                    copy_data=copy_data,
                    drop_last=True,
                    mode="val",
                    fold=f)

                _, test_loader, test_sampler = make_hospital(
                    transform=transform,
                    batch_size=batch_size,
                    # collator=mask_collator,
                    pin_mem=pin_mem,
                    training=False,
                    num_workers=num_workers,
                    world_size=world_size,
                    rank=rank,
                    root_path=root_path,
                    image_folder=image_folder,
                    copy_data=copy_data,
                    drop_last=True,
                    mode='test',
                    fold=f)

        ipe = len(train_loader)

        encoder = DistributedDataParallel(encoder, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=True)
        target_encoder = DistributedDataParallel(target_encoder)
        # encoder.eval()

        #freeze encoder
        # for p in encoder.parameters():
        #     p.requires_grad = False

        # -- init classifier
        classifier = AttentiveClassifier(
            embed_dim=encoder.module.embed_dim,
            num_heads=encoder.module.num_heads,
            depth=1,
            num_classes=num_classes
        ).to(device)

        # -- init optimizer and scheduler
        optimizer, scaler, scheduler, wd_scheduler = init_opt_ft(
            classifier=classifier,
            wd=wd,
            final_wd=final_wd,
            start_lr=start_lr,
            ref_lr=lr,
            final_lr=final_lr,
            iterations_per_epoch=ipe,
            warmup=warmup,
            num_epochs=num_epochs,
            use_bfloat16=use_bfloat16)

        # -- momentum schedule
        momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                            for i in range(int(ipe*num_epochs*ipe_scale)+1))

        
        # -- load training checkpoint
        if load_model:
            #TODO: we only need the encoder for finetuning?
            encoder, predictor, target_encoder, optimizer, scaler, _ = load_checkpoint_ft(
                device=device,
                r_path=load_path,
                encoder=encoder,
                predictor=predictor,
                target_encoder=target_encoder,
                opt=optimizer,
                scaler=scaler)
            # #TODO: this is the for continue training, so should be deleted?
            # for _ in range(start_epoch*ipe):
            #     scheduler.step()
            #     wd_scheduler.step()
            #     next(momentum_scheduler)
            #     mask_collator.step()

        # init the finetune model
        # model = init_finetune_model(
        #     device=device,
        #     encoder=encoder,
        #     # encoder=target_encoder,
        #     num_classes=num_classes,
        #     patch_size=patch_size, 
        #     model_name=model_name,
        #     crop_size=crop_size,
        #     in_chans=in_chans,
        # )
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        #TODO: need a new save function
        def save_checkpoint(epoch, best=False):
            save_dict = {
                'encoder': encoder.state_dict(),
                'classifier': classifier.state_dict(),
                'opt': optimizer.state_dict(),
                'scaler': None if scaler is None else scaler.state_dict(),
                'epoch': epoch,
                'loss': loss_meter.avg,
                'batch_size': batch_size,
                'world_size': world_size,
                'lr': lr
            }
            if rank == 0:
                # torch.save(save_dict, latest_path)
                if best:
                    torch.save(save_dict, best_path)
                else:
                    torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))


        # -- TRAINING LOOP
        start_epoch = 0
        if resume_checkpoint:
            encoder, classifier, optimizer, scaler, start_epoch = load_checkpoint_ft(
                device=device,
                r_path=latest_path,
                encoder=encoder,
                classifier=classifier,
                opt=optimizer,
                scaler=scaler)
            for _ in range(start_epoch*ipe):
                scheduler.step()
                wd_scheduler.step()


        niters = 0
        best_validation_loss = 1000
        
        for epoch in range(start_epoch, num_epochs):
            training = True
            logger.info('Epoch %d' % (epoch + 1))

            # -- update distributed-data-loader epoch
            train_sampler.set_epoch(epoch)
            validation_sampler.set_epoch(epoch)

            loss_meter = AverageMeter()
            val_loss_meter = AverageMeter()
            time_meter = AverageMeter()

            classifier.train(mode=training)
            criterion = torch.nn.CrossEntropyLoss()
            #

            for itr, batch_data in enumerate(train_loader):
                imgs = batch_data[0].to(device, non_blocking=True)
                target = batch_data[1].to(device, non_blocking=True)

                def train_step():
                    _new_lr = scheduler.step()
                    _new_wd = wd_scheduler.step()
                    # --
                    #TODO: change forward fucntion and loss_fn
                    # Step 1. Forward
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                        # outputs = F.softmax(model(imgs), dim=1)
                        # loss = model.criterion(outputs, target)
                        # with torch.no_grad():
                        #     outputs = encoder(imgs)
                        outputs = encoder(imgs)
                        outputs = classifier(outputs)
                    loss = criterion(outputs, target)

                    #  Step 2. Backward & step
                    if use_bfloat16:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    grad_stats = grad_logger(encoder.named_parameters())
                    optimizer.zero_grad()

                    return (float(loss), _new_lr, _new_wd, grad_stats)

                (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
                loss_meter.update(loss)
                time_meter.update(etime)

                # -- Logging
                def log_stats():
                    csv_logger.log(epoch + 1, itr, loss, etime)
                    if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                        logger.info('[%d, %5d] loss: %.3f '
                                    '[wd: %.2e] [lr: %.2e] '
                                    '[mem: %.2e] '
                                    '(%.1f ms)'
                                    % (epoch + 1, itr,
                                    loss_meter.avg,
                                    _new_wd,
                                    _new_lr,
                                    torch.cuda.max_memory_allocated() / 1024.**2,
                                    time_meter.avg))

                        if grad_stats is not None:
                            logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                        % (epoch + 1, itr,
                                        grad_stats.first_layer,
                                        grad_stats.last_layer,
                                        grad_stats.min,
                                        grad_stats.max))

                log_stats()

                assert not np.isnan(loss), 'loss is nan'

                if rank == 0 and not disable_wandb:
                    wandb.log(
                        {
                        f"Fold {f} - lr": _new_lr,
                        f"Fold {f} - Training Loss": loss,
                        "custom_step": niters,
                        },
                        # step=niters,
                    )
                niters += 1

            logger.info('avg. loss %.3f' % loss_meter.avg)

            #iterate over validation dataloader
            training = False
            classifier.train(mode=training)
            for itr, batch_data in enumerate(validation_loader):

                imgs = batch_data[0].to(device, non_blocking=True)
                target = batch_data[1].to(device, non_blocking=True)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    # outputs = F.softmax(model(imgs), dim=1)
                    # loss = model.criterion(outputs, target)
                    with torch.no_grad():
                        outputs = encoder(imgs)
                        outputs = classifier(outputs)
                loss = criterion(outputs, target)    
                val_loss_meter.update(float(loss))

                assert not np.isnan(float(loss)), 'loss is nan'

            #log
            if rank == 0 and not disable_wandb:
                wandb.log(
                    {
                    f"Fold {f} - Validation Loss": val_loss_meter.avg,
                    "custom_step": niters,
                    },
                    # step=niters,
                )

            if (epoch != 0) and (val_loss_meter.avg < best_validation_loss):
                best_validation_loss=val_loss_meter.avg
                save_checkpoint(epoch+1, best=True)

            if (epoch + 1) % checkpoint_freq == 0:
                save_checkpoint(epoch+1)
        
        #TESTING
        Training=False
        classifier.train(mode=training)
        test_losses = []
        predictions = []
        label_test = []
        #iterate over validation dataloader
        for itr, batch_data in enumerate(test_loader):
            imgs = batch_data[0].to(device, non_blocking=True)
            target = batch_data[1].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                # outputs = F.softmax(model(imgs), dim=1)
                # predictions.extend(torch.argmax(outputs, dim=1).tolist())
                # label_test.extend(torch.argmax(target, dim=1).tolist())
                # loss = model.criterion(outputs, target)
                with torch.no_grad():
                    outputs = encoder(imgs)
                    outputs = classifier(outputs)
                predictions.extend(torch.argmax(outputs, dim=1).tolist())
                label_test.extend(torch.argmax(target, dim=1).tolist())

            loss = criterion(outputs, target)    
            test_losses.append(float(loss))

        accuracy = accuracy_score(label_test, predictions)
        bal_acc = balanced_accuracy_score(label_test, predictions)
        precision = precision_score(label_test, predictions, average='macro')
        recall = recall_score(label_test, predictions, average='macro')
        f1 = f1_score(label_test, predictions, average='macro')

        fivefolds_test_loss.append(np.average(test_losses))
        fivefolds_test_accuracy.append(accuracy)
        fivefolds_test_bal_accuracy.append(bal_acc)
        fivefold_test_precision.append(precision)
        fivefold_test_recall.append(recall)
        fivefold_test_f1.append(f1)

        log_string = f" ===> Test loss: {np.average(test_losses):.05f} \n "
        log_string += f"===> Accuracy: {accuracy:.05f} \n "
        log_string += f"===> Balanced Accuracy: {bal_acc:.05f} \n "
        log_string += f"===> Precision: {precision:.05f} \n "
        log_string += f"===> Recall: {recall:.05f} \n "
        log_string += f"===> F1-score: {f1:.05f} \n "

        print(log_string)
    
    print(f"Average Test Loss: {np.mean(fivefolds_test_loss)}, std: {np.std(fivefolds_test_loss)}")
    print(f"Average Test Accuracy: {np.mean(fivefolds_test_accuracy)}, std: {np.std(fivefolds_test_accuracy)}")
    print(f"Average Test Balanced Accuracy: {np.mean(fivefolds_test_bal_accuracy)}, std: {np.std(fivefolds_test_bal_accuracy)}")
    print(f"Average Test Precision: {np.mean(fivefold_test_precision)}, std: {np.std(fivefold_test_precision)}")
    print(f"Average Test Recall: {np.mean(fivefold_test_recall)}, std: {np.std(fivefold_test_recall)}")
    print(f"Average Test F1: {np.mean(fivefold_test_f1)}, std: {np.std(fivefold_test_f1)}")

    if rank == 0 and not disable_wandb:
        run.finish()

if __name__ == "__main__":
    main()
