import yaml
import torch
import argparse
import pytorch_lightning as pl

from pl_model import LitModel
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def get_parser():
    parser = argparse.ArgumentParser(description='Tooth Landmark Detection')
    parser.add_argument("--config", type=str, default="config/keypoint_saliency.yaml", help="path to config file")
    parser.add_argument("--gpus", type=int, default=1)

    args_cfg = parser.parse_args()
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg


if __name__ == "__main__":
    args = get_parser()
    pl.seed_everything(args.seed)

    print("[bartu debug] Creating model..")
    model = LitModel(args)
    print("[bartu debug] Model created.")

    if args.load_from_checkpoint:
        model = LitModel.load_from_checkpoint(args.load_from_checkpoint)

    logger = TensorBoardLogger("runs", args.experiment)
    callback = ModelCheckpoint(monitor='val_loss', save_top_k=5, save_last=True, mode='min')

    debug = False
    debug_args = {'limit_train_batches': 10} if debug else {}

    if  torch.accelerator.is_available():
        print('[bartu debug] accelerator is avaliable!')
        device = 'cuda'
    else: 
        print('[bartu debug] cannot find an accelerator, falling back to cpu.')
        device = 'cpu'

    trainer = pl.Trainer(logger=logger, accelerator=device, devices=1, max_epochs=args.max_epochs, callbacks=[callback],
                         **debug_args)

    trainer.fit(model)

    # Bartu edit
    # based on https://github.com/m-bain/whisperX/issues/1304
    _original_torch_load = torch.load

    def _trusted_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _trusted_load
    # End of bartu edit

    results = trainer.test(ckpt_path='best')
    print(results)
