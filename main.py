import torch
model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
model = model.cuda()


image = torch.randn(1, 3, 800, 800)
image = image.cuda()

pred_depth, confidence, output_dict = model.inference({'input': image})  # (1, 1, 256, 256)
a = 1