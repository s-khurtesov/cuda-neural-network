#pragma once

class LayerShape {
public:
	int batch_size;
	int in_nrns;
	int in_nrn_h;
	int in_nrn_w;
	int out_nrns;
	int out_nrn_h;
	int out_nrn_w;

	LayerShape() : batch_size(0), in_nrns(0), in_nrn_h(0), in_nrn_w(0), out_nrns(0), out_nrn_h(0), out_nrn_w(0) {}

	LayerShape(int batch_size_, int in_nrns_, int out_nrns_, int in_nrn_h_ = 1, int in_nrn_w_ = 1, int out_nrn_h_ = 1, int out_nrn_w_ = 1)
		: batch_size(batch_size_), in_nrns(in_nrns_), in_nrn_h(in_nrn_h_), in_nrn_w(in_nrn_w_), out_nrns(out_nrns_),
		out_nrn_h(out_nrn_h_), out_nrn_w(out_nrn_w_) {}
};
