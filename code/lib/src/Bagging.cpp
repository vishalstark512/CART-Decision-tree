/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission.
 * Written by Pieter Robberechts, 2019
 */

#include "Bagging.hpp"
#include "TreeTest.hpp"

using std::make_shared;
using std::shared_ptr;
using std::string;
using boost::timer::cpu_timer;

using namespace std;

Bagging::Bagging( DataReader& dr,  int ensembleSize, int seed) :
    dr_(dr),
    ensembleSize_(ensembleSize),
    learners_({}) {
    random_number_generator.seed(seed);
    buildBag();
}


void Bagging::buildBag()
{
    cpu_timer timer;
    std::vector<double> timings;
    for (int i = 0; i < ensembleSize_; i++)
    {
        timer.start();
        DecisionTree dt(dr_, 1);
        //DecisionTree* dt = new DecisionTree(dr_, 1);
        learners_.emplace_back(dt);
    //////////////////////////////////////

    auto nanoseconds = boost::chrono::nanoseconds(timer.elapsed().wall);
    auto seconds = boost::chrono::duration_cast<boost::chrono::seconds>(nanoseconds);
    timings.push_back(seconds.count());
    }

    float avg_timing = Utils::iterators::average(std::begin(timings), std::begin(timings) + std::min(5, ensembleSize_));
    std::cout << "Average timing: " << avg_timing << std::endl;
}

void Bagging::test()  {
    TreeTest t;
    float accuracy = 0;
    for (const auto& row : dr_.testData()) {
        static size_t last = row.size() - 1;
        std::vector<std::string> decisions;
        for (int i = 0; i < ensembleSize_; i++) {
             std::shared_ptr<Node> root = std::make_shared<Node>(learners_.at(i).root_);
             auto& classification = t.classify(row, root);
            decisions.push_back(Utils::tree::getMax(classification));
        }
        std::string prediction = Utils::iterators::mostCommon(decisions.begin(), decisions.end());
        if (prediction == row[last])
            accuracy += 1;
    }
    std::cout << "Total accuracy: " << (accuracy / dr_.testData().size()) << std::endl;
}


