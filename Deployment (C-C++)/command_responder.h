/* Copyright 2022 The TensorFlow Authors. All Rights Reserved. */

// this is just the output provider

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_COMMAND_RESPONDER_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_COMMAND_RESPONDER_H_

#include "tensorflow/lite/c/common.h"

// Called every time the results of an audio recognition run are available. The
// human-readable name of any recognized command is in the `found_command`
// `score` has the numerical confidence, and `is_new_command` is set
// if the previous command was different to this one.
void RespondToCommand(int32_t current_time, const char* found_command,
                      uint8_t score, bool is_new_command);

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_COMMAND_RESPONDER_H_
