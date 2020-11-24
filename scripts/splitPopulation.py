#!/usr/bin/env python3

import pandas as pd
import argparse
import math
from pprint import pprint

def getPopulationStats(df):
    stats={}
    total=len(df)
    for name in df.species.unique():
        count = len(df[df.species==name])
        stats[name] = (count, count/total)
    return stats

if __name__=="__main__":
    """ Split a totalPopulation.csv into annotations.csv, validation.csv, and test.csv """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", help='input file to split')
    parser.add_argument('--train-csv', type=str, default='annotations.csv')
    parser.add_argument('--val-csv', type=str, default='validation.csv')
    parser.add_argument('--test-csv', type=str, default='test.csv')
    parser.add_argument('--train-pop', type=float, default=0.8)
    parser.add_argument('--val-pop', type=float, default=0.1)
    parser.add_argument('--test-pop', type=float, default=0.1)
    parser.add_argument('--random-seed', type=int, default=200)
    args = parser.parse_args()

    print(f"Train: {args.train_pop}, Val: {args.val_pop}, Test: {args.test_pop}")
    print(f"Random seed: {args.random_seed}")
    if not math.isclose(args.train_pop + args.val_pop + args.test_pop, 1.0):
        print("ERROR: population vector must sum to 1.0")
    

    cols=['img', 'x1','x2','y1','y2','species']
    full_population = pd.read_csv(args.input_csv, header=None, names=cols)

    # generate test hold out set first
    train_val_pop = 1.0 - args.test_pop
    train_images = full_population.sample(frac=train_val_pop, random_state=args.random_seed)
    train_val_df = full_population.take(train_images.index)
    test_df = full_population.drop(train_images.index)

    # make sure no test images are cross polinated with train images
    for image in test_df.img.unique():
        train_matches = train_val_df.loc[train_val_df.img == image]
        train_val_df = train_val_df.drop(train_matches.index)
        test_df = test_df.append(train_matches)
        count = len(train_matches)

    test_df.to_csv(args.test_csv, header=False, index=False)

    # generate train / validation sets on remaining data
    # account for percentages to be across the whole population
    # Crank the random number generator by 1
    train_pop = 1.0 - (args.val_pop / train_val_pop)
    train_df = train_val_df.sample(frac=train_pop, random_state=args.random_seed+1)
    val_df = train_val_df.drop(train_df.index)


    # make sure no test images are cross polinated with train images                                                                                                                    
    for	image in val_df.img.unique():
        train_matches =	train_df.loc[train_df.img == image]
        train_df = train_df.drop(train_matches.index)
        val_df = val_df.append(train_matches)
        count = len(train_matches)
    
    train_df.to_csv(args.train_csv, header=False,index=False)
    val_df.to_csv(args.val_csv, header=False, index=False)

    train_perc = len(train_df) / len(full_population)
    validation_perc = len(val_df) / len(full_population)
    test_perc = len(test_df) / len(full_population)
    
    print(f"Train Population: {train_perc:.2f}")
    pprint(getPopulationStats(train_df))

    print(f"Validation Population: {validation_perc:.2f}")
    pprint(getPopulationStats(val_df))

    print(f"Test Population: {test_perc}")
    pprint(getPopulationStats(test_df))
   
