/* Copyright 2022 The TensorFlow Authors. All Rights Reserved. */

#if defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)
#define ARDUINO_EXCLUDE_CODE
#endif  // defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)

#ifndef ARDUINO_EXCLUDE_CODE

#include "Arduino.h"
#include "command_responder.h"
#include "tensorflow/lite/micro/micro_log.h"

// Serial prints the output. I'm planning to link this up to a screen
// and make it show the prediction over the screen...
void RespondToCommand(int32_t current_time, const char* found_command,
                      uint8_t score, bool is_new_command) {
  if (is_new_command) {
    MicroPrintf("Heard: %s (%d), Time: %d ms", found_command, score, current_time);
  }
}

#endif  // ARDUINO_EXCLUDE_CODE
