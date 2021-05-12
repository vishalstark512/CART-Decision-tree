/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission. 
 * Written by Pieter Robberechts, 2019
 */

#include <cmath>
#include <algorithm>
#include <iterator>
#include "Calculations.hpp"
#include "Utils.hpp"
#include <math.h>

#include<iostream> 
#include<vector>
#include <unordered_map>
#include <map>
using namespace std;
#include <list>
#include <numeric>
#include <random>
#include <string>
#include <sstream>
#include <tuple>


using std::tuple;
using std::pair;
using std::forward_as_tuple;
using std::vector;
using std::string;
using std::unordered_map;

tuple<const Data, const Data> Calculations::partition(const Data& data, const Question& q) {
  Data true_rows;
  Data false_rows;
  
  for (const auto &row: data) {
    if (q.solve(row))
      true_rows.push_back(row);
    else
      false_rows.push_back(row);
  }

  return forward_as_tuple(true_rows, false_rows);
}


tuple<const double, const Question> Calculations::find_best_split(const Data& rows, const MetaData& meta) {
  double best_gain = 0.0;  // keep track of the best information gain
  auto best_question = Question();  //keep track of the feature / value that produced it
  
  //TODO: find the best split among all features and feature values
  std::string best_thresh;
  double  best_loss, gain, gini_total;
  ClassCounter counts_total;

  // calculating the class count on the parent node and calculating gini of the parent node
  counts_total = classCounts(rows);
  gini_total = gini(counts_total, rows.size());

  
  for (int i = 0; i < int(meta.labels.size()-1); i++) 
  {
      // We had additional Data stored in Meta, so when according to the feature (numerical or categorical) 
      //best threshold function os called
    if (meta.label_type[i] == "numeric") 
    {
        std::tie(best_thresh, best_loss) = determine_best_threshold_numeric(rows, meta, i);
    } else{
        std::tie(best_thresh, best_loss) = determine_best_threshold_cat(rows, meta, i);
    }

    // from above functions we get the best best loss that is used find the best gain
    // at the end we wget best gain, which is basically feature to chose to make it a node
    //best_question holds best threshold and the index of the best feature
    gain = gini_total - best_loss;
    if (best_gain < gain) 
    {
        best_gain = gain;
        best_question = Question(i,best_thresh);
    }
  }

  return forward_as_tuple(best_gain, best_question);
}

const double Calculations::gini(const ClassCounter& counts, double N) {
  double impurity = 1.0;

  //TODO: compute gini index, given class counts and the dataset size
  for( const auto& count : counts ) {
    impurity -= std::pow(count.second/N,2);
  }
  
  return impurity;
}



tuple<std::string, double> Calculations::determine_best_threshold_numeric(const Data& data, const MetaData& meta, int col) {
  double best_loss = 20; // Loss is always between 0 to 1, so we start best_loss with value of 20 which is baiscally infiinity in this case
  std::string best_thresh;

  double data_size = data.size();
  double leftSize = 0.0;
  double rightSize = 0.0;
  double loss;

  // this is to store the count of label for left and right subtree
  ClassCounter counts_left, counts_right;
  
  //labelCounter stores the counter of labels for each unique value in a feature
  std::unordered_map<std::string, ClassCounter> labelCounter;
  
  labelCounter = labelCounts(data, col);
  counts_right = labelCounter.at("total");

  //just storing the total elements in the right subtree
  for (const auto& rightclass : counts_right){
      rightSize +=rightclass.second;
  }

  //meta.meta.unqiue_data[col] contains all the unique value for that particular column
  for (const auto& val: meta.unqiue_data[col]) {
    //checking whether the val is present in labelCounter vector
      
    if (labelCounter.find(val) != std::end(labelCounter)) {

        //going to the val vector of labelcounter 
        for (const auto& valLabel : labelCounter.at(val)){
            //updating the size of total element in both left and right subtree as we parse all the unique values of a column
            leftSize  = leftSize + valLabel.second;
            rightSize = rightSize- valLabel.second;

            //updating the counter of left and right subtree
            counts_right[valLabel.first] = counts_right[valLabel.first]- valLabel.second;

            //checking if the vallabel, which conisits of element with their label count is present in in counts_true which itself is a class counter
            if (counts_left.find(valLabel.first) != std::end(counts_left)) {
                counts_left.at(valLabel.first) = counts_left.at(valLabel.first) + valLabel.second;
            } else {
                counts_left[valLabel.first] = counts_left[valLabel.first] + valLabel.second;
            }
        }

        // only calculate loss when there are both left and right subtree
        // saving the best threshold and best loss for a particular column
        if (rightSize > 0) 
        {
            //calculating the total loss
            loss = (leftSize / data_size *gini(counts_left, leftSize)) +  (rightSize / data_size * gini(counts_right, rightSize));
            if (loss < best_loss) {
                best_loss = loss;
                best_thresh = val;
            }
        }
    }
  }  
  
  return forward_as_tuple(best_thresh, best_loss);
}



tuple<std::string, double> Calculations::determine_best_threshold_cat(const Data& data, const MetaData& meta, int col) {
    double best_loss = 20; // Loss is always between 0 to 1, so we start best_loss with value of 20 which is baiscally infiinity in this case
    std::string best_thresh;

    double data_size = data.size();
    double leftSize = 0.0;
    double rightSize = 0.0;
    double loss;

    
    ClassCounter counts_left, counts_right;
    std::unordered_map<std::string, ClassCounter> labelCounter;

    labelCounter = labelCounts(data, col);

    //meta.meta.unqiue_data[col] contains all the unique value for that particular column
    for (const auto& val : meta.unqiue_data[col]) {
        //checking whether the val is present in labelCounter vector
        if (labelCounter.find(val) != std::end(labelCounter)) {
            counts_left = labelCounter.at(val);
            counts_right = labelCounter.at("total");

            //getting the size of the left and right subtree counter
            for (const auto& leftclass : counts_left) {
                leftSize += leftclass.second;
                counts_right[leftclass.first] = counts_right[leftclass.first] - leftclass.second;
            }
            for (const auto& rightclass : counts_right) {
                rightSize += rightclass.second;
            }

            // only calculate loss when there are both left and right subtree
        // saving the best threshold and best loss for a particular column
            if (rightSize > 0) 
            {
                //calculating the total loss
                loss = (leftSize / data_size * gini(counts_left, leftSize)) + (rightSize / data_size * gini(counts_right, rightSize));
                if (loss < best_loss)
                {
                    best_loss = loss;
                    best_thresh = val;
                }
            }
        }
    }
    return forward_as_tuple(best_thresh, best_loss);
}


const ClassCounter Calculations::classCounts(const Data& data) {
  ClassCounter counter;
  for (const auto& rows: data) {
    const string decision = *std::rbegin(rows);
    if (counter.find(decision) != std::end(counter)) {
      counter.at(decision)++;
    } else {
      counter[decision] += 1;
    }
  }
  return counter;
}

const std::unordered_map<std::string, ClassCounter> Calculations::labelCounts(const Data& data, int col) {
  
  /* label counter looks like this
  *
  * 4, 5, 6 ,..... are values from a particular feature and let's say we have 3 classes A, B and C
  *
  * "4": {"class A":"3"}
  * "5": {"class A":"3", "class B":"10"}
  * "8": {"class A":"9", "class C":"10"}
  */

    std::unordered_map<std::string, ClassCounter> label_counter;

  for (const auto& rows: data) 
  {
    const string val = rows[col];

    // For each element in the data calculating the label count and storing it in an unordered map
    //if val is present in label_counter then only enter this if condition
    if (label_counter.find(val) != std::end(label_counter))
    {

        // if a particular label is present in the countlabel of a prticular feature value then increase its count otherwise got to else
        // add that variable to the unordered map and give it value of 1
      if (label_counter[val].find(rows.back()) != std::end(label_counter[val]))
      {
          label_counter[val].at(rows.back())++;
      } 
      else 
      {
          label_counter[val][rows.back()] += 1;
      }

    } 

    else 
    {
        label_counter[val][rows.back()] = 1;
    }

    if (label_counter.find("total") != std::end(label_counter))
    {
      if (label_counter["total"].find(rows.back()) != std::end(label_counter["total"]))
      {
          label_counter["total"].at(rows.back())++;
      } 
      else 
      {
          label_counter["total"][rows.back()] += 1;
      }
    } 
    else 
    {
        label_counter["total"][rows.back()] = 1;
    }
  }  

  return label_counter;
}
