/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission. 
 * Written by Pieter Robberechts, 2019
 */

/*
 * Following changes were made:
 * Added getColumn to return the attritbute used in the question
 * Added an overload to solve
 */

#include "Question.hpp"
#include "Utils.hpp"

using std::string;
using std::vector;

Question::Question() : column_(0), value_("") {}

Question::Question(const int column, const string value) : column_(column), value_(value) {}

const bool Question::solve(VecS example) const {
  const string& val = example[column_];
  if (isNumeric(val)) {
    return std::stod(val) >= std::stod(value_);
  } else {
    return val == value_;
  }
}
const string Question::toString(const VecS& labels) const {
  string condition = "==";
  if (isNumeric(value_))
    condition = ">=";
  return "Is " + labels[column_] + " " + condition + " " + value_ + "?";
}

const bool Question::isNumeric(std::string value) const {
  if (bool empty = value.empty(); !empty) {
    try {
      std::stod(value);
    } catch (const std::exception &e) {
      return false;
    }
  }
  return true;
}

const bool Question::isNumeric(void) const {
  return Question::isNumeric(value_); 
}

const bool Question::solve(std::string value) const {
    if (isNumeric(value)) {
        return std::stod(value) >= std::stod(value_);
    } else {
        return value == value_;
    }
}
const int Question::getColumn() const {
    return column_;
}
