/**
 * Minimal MLP evaluator for the Structure Hold policy.
 *
 * The released actors (training/policies, .onnx) are observation normalisation followed by
 * a few dense layers with ELU, so the onboard "ONNX runtime" is this header plus the weights
 * that tools/export_policy_header.py writes into PolicyWeights.hpp (docs/decisions.md). No
 * heap, no PX4 headers, no third-party library: it compiles on NuttX and in the plain g++
 * unit test (tests/test_policy.cpp), where its output is checked against onnxruntime.
 *
 * Also here: the rolling observation history the policy consumes (oldest first), sized at
 * compile time from policy.json input.history through the generated header.
 */

#pragma once

#include <cmath>
#include <cstddef>

namespace structure_hold
{
namespace policy
{

enum class Activation { None, Elu, Relu, Tanh };

struct Layer {
	int in;
	int out;
	const float *w;   // row-major (out, in)
	const float *b;   // (out)
	Activation activation;
};

struct Net {
	int input_size;
	int output_size;
	const float *mean;  // (input_size) observation normaliser, folded in at export
	const float *std;
	const Layer *layers;
	int num_layers;
};

/**
 * Evaluate ``net`` on ``x`` (input_size floats) into ``y`` (output_size floats).
 * ``scratch`` must hold 2 * max_width floats; the caller owns it so nothing is allocated.
 * Returns false (and leaves ``y`` untouched) if the buffers are too small.
 */
inline bool forward(const Net &net, const float *x, float *y, float *scratch, int max_width)
{
	if (net.input_size > max_width || net.output_size > max_width) {
		return false;
	}

	float *cur = scratch;
	float *next = scratch + max_width;

	for (int i = 0; i < net.input_size; i++) {
		cur[i] = (x[i] - net.mean[i]) / net.std[i];
	}

	for (int l = 0; l < net.num_layers; l++) {
		const Layer &L = net.layers[l];

		if (L.in > max_width || L.out > max_width) {
			return false;
		}

		for (int o = 0; o < L.out; o++) {
			const float *row = L.w + static_cast<std::size_t>(o) * L.in;
			float acc = L.b[o];

			for (int i = 0; i < L.in; i++) {
				acc += row[i] * cur[i];
			}

			switch (L.activation) {
			case Activation::Elu: acc = acc > 0.f ? acc : expm1f(acc); break;

			case Activation::Relu: acc = acc > 0.f ? acc : 0.f; break;

			case Activation::Tanh: acc = tanhf(acc); break;

			case Activation::None: break;
			}

			next[o] = acc;
		}

		float *t = cur;
		cur = next;
		next = t;
	}

	for (int o = 0; o < net.output_size; o++) {
		y[o] = cur[o];
	}

	return true;
}

/**
 * Rolling window of the last HISTORY observations of NUM_OBS floats each, oldest first,
 * mirroring training/ppo_env.py ObsHistory: on engage the whole window is filled with the
 * first observation, then each push drops the oldest.
 */
template<int HISTORY, int NUM_OBS>
class ObsHistory
{
public:
	static constexpr int SIZE = HISTORY * NUM_OBS;

	void fill(const float *obs)
	{
		for (int h = 0; h < HISTORY; h++) {
			for (int i = 0; i < NUM_OBS; i++) {
				_buf[h * NUM_OBS + i] = obs[i];
			}
		}

		_filled = true;
	}

	void push(const float *obs)
	{
		if (!_filled) {
			fill(obs);
			return;
		}

		for (int h = 0; h + 1 < HISTORY; h++) {
			for (int i = 0; i < NUM_OBS; i++) {
				_buf[h * NUM_OBS + i] = _buf[(h + 1) * NUM_OBS + i];
			}
		}

		for (int i = 0; i < NUM_OBS; i++) {
			_buf[(HISTORY - 1) * NUM_OBS + i] = obs[i];
		}
	}

	void clear() { _filled = false; }
	bool filled() const { return _filled; }
	const float *stacked() const { return _buf; }

private:
	float _buf[SIZE] {};
	bool _filled{false};
};

} // namespace policy
} // namespace structure_hold
