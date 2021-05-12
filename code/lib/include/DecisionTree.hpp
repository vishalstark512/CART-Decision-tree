/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission.
 * Written by Pieter Robberechts, 2019
 */

#ifndef DECISIONTREE_DECISIONTREE_HPP
#define DECISIONTREE_DECISIONTREE_HPP

#include <random>
#include <boost/chrono.hpp>
#include "Calculations.hpp"
#include "DataReader.hpp"
#include "Node.hpp"
#include "TreeTest.hpp"
#include "Utils.hpp"
#include <algorithm>
#include <chrono>


class DecisionTree {
public:
    DecisionTree() = delete;
    explicit DecisionTree ( DataReader& dr, int dt_type);
    explicit DecisionTree ( DataReader& dr, const std::vector<size_t>& samples);

    void print() ;
    void test() ;

    //inline Data testData() { return dr_.testData(); }
    inline std::shared_ptr<Node> root() { return std::make_shared<Node>(root_); }

    DataReader dr_;
    Node root_;
private:
    

    const Node buildTree( Data& rows,  MetaData& meta,int dt_type);
    void print( std::shared_ptr<Node> root, std::string spacing = "") ;
    int Treedepth = 0;

};

#endif //DECISIONTREE_DECISIONTREE_HPP
