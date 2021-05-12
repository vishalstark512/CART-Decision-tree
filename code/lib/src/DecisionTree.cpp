/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission.
 * Written by Pieter Robberechts, 2019
 */

#include "DecisionTree.hpp"
#include <algorithm>
#include <chrono>
#include <time.h>

using std::make_shared;
using std::shared_ptr;
using std::string;
using boost::timer::cpu_timer;

DecisionTree::DecisionTree( DataReader& dr, int dt_type) : root_(Node()), dr_(dr) {

    std::cout << "Start building tree." << std::endl; cpu_timer timer;

    // this part is called when you train a single decision tree
    if (dt_type == 0)
    {
        root_ = buildTree(dr_.trainData(), dr_.metaData(), dt_type);
    } 

    // this part is called if you want bagging to run
    if (dt_type == 1)
    {
        // We are creating random batches of 1/70th part of data, change this if you want bigger or smaller batch
        int batch_size = dr_.trainData().size() / 70;

        // randomly generate values based on the system clock because every batch need to be different so we are changing the seed
        srand(time(NULL));
        unsigned seed = rand() % 100 + 1;
        std::default_random_engine generator(seed);
        std::uniform_int_distribution<int> distributionInteger(0, dr_.trainData().size() - 1);

        // Craeting batches of the data
        Data Batched;
        for (int a = 0; a < batch_size; a += 1)
        {
            int r = distributionInteger(generator);
            Batched.push_back(std::vector<string>());
            for (int b = 0; b < dr_.trainData()[r].size(); b++)
            {
                Batched[a].push_back(dr_.trainData()[r][b]);

            }
        }
        root_ = buildTree(Batched, dr_.metaData(), dt_type);
        std::vector<std::vector<std::string>> a{ {"a"},{"b"} };

        // deleting this is important otherwise learners vectors goes beyond memory 
        //bcz it saves entire dataframe copy in the learners vector every learner

        dr_.trainData() = a;
        dr_.testData() = a;
        //print();
        //test();
        
    }

    std::cout << "Done. " << timer.format() << std::endl;
}

const Node DecisionTree::buildTree( Data& rows,  MetaData& meta, int dt_type) {
    //TODO: build a decision tree
    double best_gain;
    Question Dt_Split;
    std::tie(best_gain, Dt_Split) = Calculations::find_best_split(rows, meta);

    
    //if we call bagging unpruned trees are trained and for single decision tree Tree are capped at the depth of 10000
    //if(best_gain < 0.001 or Treedepth>2000)
    if (best_gain == 0.0 or (dt_type == 0 and Treedepth > 20000))  //check gain or depth for leaf creation
    {
        // return the leaf node which consits the counts of each class it has in the leaf node
        return Node(Leaf(Calculations::classCounts(rows)));
    }
    else
    {
        Treedepth += 1;

        Data left_rows, right_rows;
        Node left_branch, right_branch;
        std::tie(left_rows, right_rows) = Calculations::partition(rows, Dt_Split);

        // recursively theses functions to build left and right part of the tree
        left_branch = buildTree(left_rows, meta, dt_type);
        right_branch = buildTree(right_rows, meta, dt_type);
        return Node(left_branch, right_branch, Dt_Split);
    }
}

void DecisionTree::print()  {
    print(make_shared<Node>(root_));
}

void DecisionTree::print( shared_ptr<Node> root, string spacing)  {
    if (bool is_leaf = root->leaf() != nullptr; is_leaf) {
        const auto& leaf = root->leaf();
        std::cout << spacing + "Predict: "; Utils::print::print_map(leaf->predictions());
        return;
    }
    std::cout << spacing << root->question().toString(dr_.metaData().labels) << "\n";

    std::cout << spacing << "--> True: " << "\n";
    print(root->trueBranch(), spacing + "   ");

    std::cout << spacing << "--> False: " << "\n";
    print(root->falseBranch(), spacing + "   ");
}

void DecisionTree::test()  {
    TreeTest t(dr_.testData(), dr_.metaData(), root_);
}
