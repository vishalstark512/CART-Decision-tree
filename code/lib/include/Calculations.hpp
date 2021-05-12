/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission. 
 * Written by Pieter Robberechts, 2019
 */


/*
 * Edited:
 * Input of determine_best_threshold_numeric, determine_best_threshold_cat
 * Added function partition_meta to split the possible values in the meta structure
 */

#ifndef DECISIONTREE_CALCULATIONS_HPP
#define DECISIONTREE_CALCULATIONS_HPP

#include <tuple>
#include <vector>
#include <string>
#include <unordered_map>
#include <boost/timer/timer.hpp>
#include "Question.hpp"
#include "Utils.hpp"

using namespace std;
using ClassCounter = std::unordered_map<std::string, int>;

namespace Calculations {

std::tuple<const Data, const Data> partition(const Data &data, const Question &q);
std::tuple<const MetaData, const MetaData> partition_meta(const MetaData &meta, const Question &q);
const Data sample_data(const Data& data, const int& iter, const int& batchSize);

const double gini(const ClassCounter& counts, double N);

std::tuple<const double, const Question> find_best_split(const Data &rows, const MetaData &meta);

std::tuple<std::string, double> determine_best_threshold_numeric(const Data &data, const MetaData &meta, int col);

std::tuple<std::string, double> determine_best_threshold_cat(const Data &data, const MetaData &meta, int col);

//we added this
//vector<vector<string>> transpose_Data(vector<vector<string>> &data);

const ClassCounter classCounts(const Data &data);
const std::unordered_map<std::string, ClassCounter> labelCounts(const Data& data, int col);

} // namespace Calculations

#endif //DECISIONTREE_CALCULATIONS_HPP
