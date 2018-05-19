"""
Module for generating achors for multiple scale feature maps
"""

import numpy as np


class AnchorGenerator:
	def __init__(self, feature_maps_sizes, anchors_scales, anchors_ratios, offsets, base_anchors_sizes):
		self.feature_maps_sizes = feature_maps_sizes
		self.anchors_scales = anchors_scales
		self.anchors_ratios = anchors_ratios
		self.offsets = offsets
		self.base_anchors_sizes = base_anchors_sizes

	def generate(self):
		anchors = []
		for i in range(len(self.feature_maps_sizes)):
			layers_anchors = self.generate_anchors_grid(self.feature_maps_sizes[i][0], self.feature_maps_sizes[i][1],
			                                            stride=1, offset=0, scales=self.anchors_scales[i],
			                                            ratios=self.anchors_ratios[i],
			                                            base_size=self.base_anchors_sizes[i])
			anchors.append(layers_anchors)
		return anchors

	def generate_anchors_grid(self, grid_height, grid_width, stride, offset, scales, ratios, base_size):
		y_centers = np.arange(grid_height).astype(np.float32) * stride + offset
		x_centers = np.arange(grid_width).astype(np.float32) * stride + offset
		heights = (scales / ratios) * base_size  # todo: make sure if the ratios are used as is or square rooted
		widths = (scales * ratios) * base_size
		xy_grid = np.meshgrid(x_centers, y_centers)
		wh_grid = np.meshgrid(widths, heights)
		sizes_grid = np.concatenate([xy_grid, wh_grid], axis=-1)
		corners_grid = self.sizes_to_corners(sizes_grid)
		return corners_grid

	def match_anchors_with_gt(self, anchors, gt):


	@staticmethod
	def sizes_to_corners(boxes):
		corners = np.empty(boxes.shape, dtype=np.float32)
		corners[:, :, :, 0] = boxes[:, :, :, 0] - boxes[:, :, :, 2] / 2.
		corners[:, :, :, 1] = boxes[:, :, :, 1] - boxes[:, :, :, 3] / 2.
		corners[:, :, :, 2] = boxes[:, :, :, 0] + boxes[:, :, :, 2] / 2.
		corners[:, :, :, 3] = boxes[:, :, :, 1] + boxes[:, :, :, 3] / 2.
		return corners
