// Copyright 2020 The TensorFlow Authors. All Rights Reserved.

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_MODEL_SETTINGS_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_MODEL_SETTINGS_H_

// Keeping these as constant expressions allow us to allocate fixed-sized arrays
// on the stack for our working memory.
constexpr int kAudioSampleFrequency = 16000; // sample_rate

constexpr int kMaxAudioSampleSize = 512; // number of samples in the FFT. not STFT window size.
          //  hann window duration
constexpr int kFeatureSliceDurationMs = 25; //this is NOT kMaxAudioSampleSize * kAudioSampleFrequency. 
constexpr int kFeatureSliceStrideMs = 15; 

constexpr int kFeatureSliceSize = 40; //mel_bins.
constexpr int kFeatureSliceCount = 66; //number of time frames.
constexpr int kFeatureElementCount = (kFeatureSliceSize * kFeatureSliceCount); //size of featureData buffer

// variables to create the filters matrix.
constexpr int kFTTBins = (kMaxAudioSampleSize / 2) + 1;

// Variables for the model's output categories.
constexpr int kUnknownIndex = 4;

// number of classes
constexpr int kCategoryCount = 5;
extern const char* kCategoryLabels[kCategoryCount];

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_MODEL_SETTINGS_H_
