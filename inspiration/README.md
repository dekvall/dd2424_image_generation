## Running StackGAN
Run StackGAN on GCP from the code folder with
```bash
python2 main.py --cfg cfg/coco_eval.yml --gpu 0
```
Contrary to popular belief setting `--gpu 0` here actually refers to the id of the gpu. In most other cases `gpu 0` refers to cpu mode. Weird.

The generated images will be stored in the `models/coco/netG_epoch_90` directory.
