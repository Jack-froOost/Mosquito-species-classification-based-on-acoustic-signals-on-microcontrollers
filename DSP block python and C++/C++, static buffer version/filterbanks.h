#pragma once

#define num_mel_bins 40
#define fft_bins 257
extern const float __attribute__((aligned(4))) filterbanks[num_mel_bins][fft_bins];

