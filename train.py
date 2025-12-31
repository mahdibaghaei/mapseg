from cfg.default import get_cfg_defaults
# import wandb
import torch
import numpy as np
from model.solver import get_solver
from data.get_dataloader import get_dataloader
import time
import os
import shutil
import yaml
import argparse
from datetime import datetime
from model.utils.util import gen_code_archive
import torch.backends.cudnn as cudnn
import random

def set_random_seed(seed):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

parser = argparse.ArgumentParser(description='Setting config file')
parser.add_argument('--config', type=str, required=True,
                    help='path to the config yaml file')
args = parser.parse_args()

if __name__ == '__main__':
    '''
    Initialize the config file
    '''
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    print('loaded configuration file {}'.format(args.config))
    cfg.freeze()

    if cfg.system.seed is not None:
        set_random_seed(cfg.system.seed)

    '''
    Initialize the checkpoint folder and save the config file
    '''
    ckpt_fld = os.path.join(cfg.system.ckpt_dir,
                            cfg.system.project, cfg.system.exp_name)
    if not os.path.exists(ckpt_fld):
        os.makedirs(ckpt_fld)

    if not os.path.exists(os.path.join(ckpt_fld, 'train_cfg.yaml')):
        with open(os.path.join(ckpt_fld, 'train_cfg.yaml'), 'w') as f:
            f.write(cfg.dump())
            f.close()
    else:
        time_now = datetime.now()
        cfg_fname = os.path.join(
            ckpt_fld, 'train_cfg_' + time_now.strftime('%Y%m%d_%H%M%S') + '.yaml')
        with open(cfg_fname, 'w') as f:
            f.write(cfg.dump())
            f.close()

    '''
    GPU Setup and Diagnostics
    '''
    print("\n" + "="*60)
    print("GPU SETUP")
    print("="*60)
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
    print("="*60 + "\n")

    '''
    Get the solver and dataloader
    '''
    cudnn.benchmark = True

    # Create solver FIRST
    train_solver = get_solver(cfg)

    # Check for existing checkpoint BEFORE wrapping with DataParallel
    checkpoint_loaded = False
    if os.path.exists(os.path.join(ckpt_fld, 'solver_latest.pth')):
        print('Found existing checkpoint, loading...')
        train_solver = torch.load(os.path.join(ckpt_fld, 'solver_latest.pth'))
        print('Loaded the latest solver checkpoint')
        print('Previous epochs: ' + str(train_solver._get_epoch()))
        checkpoint_loaded = True

    # Now apply DataParallel if multiple GPUs and NOT loaded from checkpoint
    # (checkpoints already have DataParallel if they were saved with it)
    if num_gpus > 1 and not checkpoint_loaded:
        print(f"\n[Multi-GPU] Wrapping model with DataParallel for {num_gpus} GPUs")
        train_solver.model = torch.nn.DataParallel(
            train_solver.model,
            device_ids=list(range(num_gpus))
        )
        effective_batch = cfg.train.batch_size * num_gpus
        print(f"[Multi-GPU] Batch size per GPU: {cfg.train.batch_size}")
        print(f"[Multi-GPU] Effective total batch size: {effective_batch}")
        print("[Multi-GPU] Both GPUs will be utilized\n")
    elif num_gpus > 1 and checkpoint_loaded:
        print(f"[Multi-GPU] Checkpoint loaded - DataParallel state preserved")
        if isinstance(train_solver.model, torch.nn.DataParallel):
            print(f"[Multi-GPU] Model is already wrapped with DataParallel")
    else:
        print("[Single-GPU] Using 1 GPU\n")

    # initialize the wandb
    if cfg.system.wandb:
        wandb.init(project=cfg.system.project,
                   config=cfg, name=cfg.system.exp_name)

    # get data loader
    train_loader = get_dataloader(cfg, train=True)

    '''
    set up validation parameters
    '''
    run_val = False
    if cfg.train.type == 'mpl':
        run_val = True

    start_epoch = 1 + train_solver._get_epoch()

    # save the current code
    gen_code_archive(ckpt_fld)

    print('Start training with this config:')
    print(cfg)

    # Clear GPU cache before training
    torch.cuda.empty_cache()

    for epoch in range(start_epoch, cfg.train.niter + cfg.train.niter_decay+1):
        epoch_start_time = time.time()
        print_start_time = time.time()

        # training
        # first to initialize the internal log of loss
        train_solver._init_epoch()

        for i, data in enumerate(train_loader):
            train_solver.train_step(data, epoch)

            if i % cfg.train.print_freq == 0:
                # step-wise log into wandb was disabled because it is too messy
                # one can enable it if needed
                # if cfg.system.wandb:
                #     wandb.log(
                #         {k+'_steps': v for k, v in train_solver.get_cur_loss().items()})
                train_solver.print_cur_loss(epoch, i, print_start_time)
                print_start_time = time.time()

                # Print GPU memory usage periodically
                if i % (cfg.train.print_freq * 5) == 0 and num_gpus > 1:
                    print(f"GPU Memory Status:")
                    for gpu_id in range(num_gpus):
                        mem_alloc = torch.cuda.memory_allocated(gpu_id) / 1024**3
                        mem_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                        print(f"  GPU {gpu_id}: {mem_alloc:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

        # summarize this epoch's results
        train_solver._log_internal_epoch_res(len(train_loader))

        if cfg.system.wandb:
            wandb.log(
                {k+'_epoch': v for k, v in train_solver._get_internal_loss().items()})

        # validation
        if run_val:
            save_best = train_solver.validation(epoch)
            if cfg.system.wandb:
                wandb.log({'validation dice': train_solver.val_dice[-1],
                          'validation score': train_solver.val_score[-1]})
            print(
                f"Epoch: {epoch}, Validation Dice: {train_solver.val_dice[-1]}, Validation score: {train_solver.val_score[-1]}, target pseudo loss: {train_solver.tgt_pse_seg_loss[-1]}")

            if save_best:
                # Save model state dict properly
                if isinstance(train_solver.model, torch.nn.DataParallel):
                    torch.save(train_solver.model.module.state_dict(),
                              os.path.join(ckpt_fld, 'best_model.pth'))
                else:
                    torch.save(train_solver.model.state_dict(),
                              os.path.join(ckpt_fld, 'best_model.pth'))

            print('Current cumulative epochs of no improvement: ' +
                  str(train_solver.cumulative_no_improve[-1]))
            if train_solver.cumulative_no_improve[-1] > cfg.train.patience:
                print('Early stopping')
                break

        # get the visualization
        train_solver.save_visualization(epoch)

        # save the model
        if epoch % cfg.train.save_epoch_freq == 0:
            if isinstance(train_solver.model, torch.nn.DataParallel):
                torch.save(train_solver.model.module.state_dict(),
                          os.path.join(ckpt_fld, f'model_epoch_{epoch}.pth'))
            else:
                torch.save(train_solver.model.state_dict(),
                          os.path.join(ckpt_fld, f'model_epoch_{epoch}.pth'))

        # save the latest solver status:
        torch.save(train_solver,
                   os.path.join(ckpt_fld, 'solver_latest.pth'))

        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, cfg.train.niter + cfg.train.niter_decay, time.time() - epoch_start_time))

        if epoch > cfg.train.niter:
            train_solver.scheduler_step()
            if cfg.system.wandb:
                wandb.log({'lr': train_solver.optimizer.param_groups[0]['lr']})

        if cfg.system.wandb:
            wandb.log({'epoch': epoch})

        # Clear cache after each epoch
        if epoch % 5 == 0:
            torch.cuda.empty_cache()

    # Save final model
    if isinstance(train_solver.model, torch.nn.DataParallel):
        torch.save(train_solver.model.module.state_dict(),
                  os.path.join(ckpt_fld, 'model_final.pth'))
    else:
        torch.save(train_solver.model.state_dict(),
                  os.path.join(ckpt_fld, 'model_final.pth'))
