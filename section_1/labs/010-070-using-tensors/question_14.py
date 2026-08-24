"""
Change the image tensor layout from NCHW to NHWC and create a contiguous copy. Print the new shape and contiguous status.

NOTE: Reorder the dimensions in the order 0, 2, 3, 1 before making the tensor contiguous.
"""
import torch

images = torch.rand(2, 3, 8, 8)
channel_last = images.____(____, ____, ____, ____)
contiguous_channel_last = channel_last.____()
print(channel_last.shape, contiguous_channel_last.is_contiguous())
