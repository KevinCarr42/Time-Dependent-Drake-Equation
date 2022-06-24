import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.special import gamma  # not the one from scipy.stats
from math import factorial, exp, log
import datetime, time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import seaborn as sns


### Parameters, Constants, and Pseudo-constants
    
def params_set(default_params):
    """
    resets params to default
    """
    return default_params.copy()


def params_rand(random_dict_of_lists):
    """
    randomises params
    """
    output_dict = {}
    for key in random_dict_of_lists:
        output_dict[key] = np.random.choice(random_dict_of_lists[key])
    
    return output_dict
    

### The Original Drake Equation
def Drake(params):
    """
    takes in dict of params, uses the params needed, calculates The Drake Equation
    The Drake Equation
    N = number of civilizations with which humans could communicate
    Rs = mean rate of star formation
    fp = fraction of stars that have planets
    ne = mean number of planets that could support life per star with planets
    fl = fraction of life-supporting planets that develop life
    fi = fraction of planets with life where life develops intelligence
    fc = fraction of intelligent civilizations that develop communication
    L = mean length of time that civilizations can communicate
    """
    return round(params['RS'] * params['FP'] * params['NE'] * params['FL'] * params['FI'] * params['FC'] * params['L'])


### Custom Functions ###

# number of stars
def star_formation(params, current_year, num_stars):
    """
    star birth
    currently 30 times lower than at the start of the universe
    peaked 8 billion years ago
    approx rate calc assumes 30x at 10B ago, and Rs from now forward, linear interpolation
    
    star death
    not based on astro physics, just used to balance the births vs death at num_galaxy
    """
    # star birth
    rate = params['RS'] * max(((params['MODERN_ERA'] - current_year) * 30) / params['MODERN_ERA'], 1)
    birth = rate * params['YEAR_STEPS']
    
    # star death
    death = rate * params['YEAR_STEPS'] * num_stars / params['NUM_GALAXY']
    
    return birth - death


# number of new planets
def new_planets(params, num_new_stars):
    """
    New stars create new planets. These planets are not habitable yet, but 
    they only count if they will become habitable, and will develop life.
    
    Incorporates all of the following Drake parameters:
    fp = fraction of stars that have planets
    ne = mean number of planets that could support life per star with planets
    fl = fraction of life-supporting planets that develop life
    """
    return num_new_stars * params['FP'] * params['NE'] * params['FL']


# number of newly habitable planets
def new_habitable(params, num_planets, num_habitable_planets):
    """
    the number of planets that become habitable
    doesn't track planets, just uses ratios to approximate
    """
    uninhabited = num_planets - num_habitable_planets
    newly_habitable = uninhabited * params['YEAR_STEPS'] / params['YEARS_PLANETS_TO_HABITABLE']
    return newly_habitable


# probability calcs whether something happens over a period
def prob_poisson(avg_time_to_happen, how_many_years_happened):
    """
    use poisson probably
    returns a ratio between 0 and 1
    results work for better known calcs, but fall apart for L
    probability of L is calculated in prob_L()
    """
    return 1 - poisson.cdf(k=1, mu=how_many_years_happened/avg_time_to_happen)


# need a different probability distribution for L; if L may be between 
# hundreds and millions of years, 0.0000000000000000% is not realistic 
# for 1M yr survival. Weibull distributions are simple and easy to tune:
# https://en.wikipedia.org/wiki/Weibull_distribution

def prob_L(params, how_many_years_happened):
    """
    probability of technological life being extinct after how_many_years_happened
    weibull distribution
    constants from the CONSTANTS section above
    """
    
    return 1 - exp(-(how_many_years_happened/params['WEIBULL_SCALE_PARAMETER'])**params['WEIBULL_SHAPE_PARAMETER'])
    

def transition(params, num_from, num_to, prob_of_transition):
    """
    the number of previous stage that develop into new stage
        eg, the number of habitable planets that evolve life
    doesn't track planets, just uses ratios to approximate expectations
    subtracts already transitioned life-stages
    doesn't apply to technological species - treated differently
    """
    return (num_from - num_to) * prob_poisson(prob_of_transition, params['YEAR_STEPS'])
    

def new_technological(params, num_cultural_life, N):
    """
    the number of planets with cultural life that develop technological life
    doesn't track planets, uses ratios to approximate expectations
    doesn't account for number of tech species, ie, doesn't use (num_cultural_life - num_tech_life)
    this is because of P_tech_dominance:
        some proportion of cultural species transition or otherwise become
        extinct as a result of sharing a planet with a technological species
        P_tech_dominance is not a CONSTANT, it is treated as an input to TimeDependentDrake()
    """
    
    return num_cultural_life * prob_poisson(params['YEARS_CULTURE_TO_TECH'], params['YEAR_STEPS'])


def new_extinctions(params, num_stars, num_life, num_complex_life, num_intelligent_life, num_cultural_life, N):
    """
    returns number of extinctions
    uses estimates for extinction events to subtract lifeforms
    should balance with star death
    """
    # base extinction rate from star death (RS and star death balance)
    base = params['RS'] * params['YEAR_STEPS'] / (num_stars + 1)  # stars plus 1 to eliminate div/0 error
    
    # specific extinction numbers
    extinction_simple = num_life * base
    extinction_complex = num_complex_life * (base + prob_poisson(params['EXTINCTION_COMPLEX'], params['YEAR_STEPS']))
    extinction_intelligent = num_intelligent_life * (base + prob_poisson(params['EXTINCTION_INTELLIGENT'], params['YEAR_STEPS']))
    extinction_cultural = num_cultural_life * (base + prob_poisson(params['EXTINCTION_CULTURAL'], params['YEAR_STEPS']))
    extinction_technological = N * prob_L(params, params['YEAR_STEPS'])
    
    return extinction_simple, extinction_complex, extinction_intelligent, extinction_cultural, extinction_technological


# The Time Dependent Drake Equation
def TimeDependentDrake(params, output_year, P_tech_dominance, df_input="empty"):
    """
    output_year is the years since 2nd gen stars, 10B ~ now
    P_tech_dominance is the proportion of intelligent lifeforms that go extinct when technological life emerges
        P_tech_dominance = 1 assumes that only one intelligent lifeform can exist once a species gains technology
    outputs the number of instances of each category active during a time step (ie, over 1M years)
    NOTE: I would be surprised if df_input="empty" is not bad form, but I'm not sure how to do this...
    """
    
    # proportion of cultural civilizations that are consumed when tech life emerges
    P_tech_dominance = P_tech_dominance
    
    # columns used
    index = 'year'
    columns = ['year', 'num_stars', 'num_planets', 'num_habitable_planets', 'num_life', 
           'num_complex_life', 'num_intelligent_life', 'num_cultural_life', 'N', 'N_extinct']
    
    # if there is an input dataframe, use that, otherwise, initialise at year 0
    if type(df_input) == pd.DataFrame:
        # start over at the last row in the input DataFrame
        history_of_life = df_input.iloc[[-1]]
    else:
        # start at year 0 with no life and no 2nd gen stars
        history_of_life = pd.DataFrame(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
            columns = columns,
            index=[0]
        ).set_index(index)
    
    # initialise variables with input dict
    current_year = history_of_life.index[-1]
    num_stars = history_of_life.num_stars.iloc[-1]
    num_planets = history_of_life.num_planets.iloc[-1]
    num_habitable_planets = history_of_life.num_habitable_planets.iloc[-1]
    num_life = history_of_life.num_life.iloc[-1]
    num_complex_life = history_of_life.num_complex_life.iloc[-1]
    num_intelligent_life = history_of_life.num_intelligent_life.iloc[-1]
    num_cultural_life = history_of_life.num_cultural_life.iloc[-1]
    N = history_of_life.N.iloc[-1]
    N_extinct = history_of_life.N_extinct.iloc[-1]

    while current_year < output_year:
        
        # increment year
        current_year += params['YEAR_STEPS']
        num_new_stars = star_formation(params, current_year, num_stars)
        
        # functions - reverse order so that growth is based of previous generation
        
        # number of extinctions
        extinctions = new_extinctions(params, num_stars, num_life, num_complex_life, num_intelligent_life, num_cultural_life, N)
        num_life -= extinctions[0]
        num_complex_life -= extinctions[1]
        num_intelligent_life -= extinctions[2]
        num_cultural_life -= extinctions[3]
        N -= extinctions[4]
        N_extinct += extinctions[4]  # track technological extinctions
        
        # number of technological species
        new_tech_life = new_technological(params, num_cultural_life, N)
        N += new_tech_life
        
        # number of cultural civilizations
        num_cultural_life += transition(params, num_intelligent_life, num_cultural_life, params['YEARS_INTELLIGENCE_TO_CULTURE'])  
        # assumes only (1 - P_tech_dominance) cultural civilizations once one civilization gains tech
        # equivalent to neandrethal going extinct before we gain technology (probably because of us)
        num_cultural_life -= new_tech_life * P_tech_dominance
        
        # number of intelligent life
        num_intelligent_life += transition(params, num_complex_life, num_intelligent_life, params['YEARS_COMPLEX_TO_INTELLIGENCE'])
        
        # number of complex lifeforms
        num_complex_life += transition(params, num_life, num_complex_life, params['YEARS_LIFE_TO_COMPLEX_LIFE'])

        # number of simple lifeforms
        num_life += transition(params, num_habitable_planets, num_life, params['YEARS_HABITABLE_TO_LIFE'])
    
        # number of habitable planets
        num_habitable_planets += new_habitable(params, num_planets, num_habitable_planets)
        
        # number of planets
        num_planets += new_planets(params, num_new_stars)
        
        # number of stars
        num_stars += num_new_stars
        
        # append new values to the history of life dataframe
        templist = [current_year, num_stars, num_planets, num_habitable_planets, num_life, 
                    num_complex_life, num_intelligent_life, num_cultural_life, N, N_extinct]
                
        history_of_life = history_of_life.append(
            pd.DataFrame(
                [templist], 
                columns = columns,
                index=[current_year]
            ).set_index(index)
        )
    
    return history_of_life
    

# how to interpret averages/expectations give 1M year time steps
def breakdown_by_year(params, N_1M_yr):
    """
    input number active over YEAR_STEPS (eg, 1M years)
    simulate number active at any given point in time
    uses random numbers, same Weibull coefficients as CONSTANTS section
    """
    # initialise 1M years all with 0s
    active_each_year = pd.DataFrame(np.zeros((params['YEAR_STEPS'], 1)))

    # loop trough each year and add N_1M_yr lifeforms, each with a random Weibull lifespan
    for n in range(N_1M_yr):
        # random species characteristics
        year_birth = int(np.random.randint(0, params['YEAR_STEPS']-1))
        year_death = int(year_birth + np.random.weibull(params['WEIBULL_SHAPE_PARAMETER']) * params['WEIBULL_SCALE_PARAMETER'])
        
        if year_death < params['YEAR_STEPS']:
            active_each_year.iloc[year_birth:year_death] += 1
        else:
            active_each_year.iloc[year_birth:params['YEAR_STEPS']] += 1
            year_death = min(year_death - params['YEAR_STEPS'], year_birth)
            active_each_year.iloc[0:year_death] += 1
        
    return active_each_year


def percentage_table(active_each_year_dataframe):
    N_active = active_each_year_dataframe.value_counts().astype('float')
    N_active_percent = (N_active / 10000).sort_index()
    print(N_active_percent.map('{:,.2f}%'.format))


def plot_histogram(active_each_year_dataframe, binwidth=1, **kwargs):
    # histogram of active technological civilizations
    plt.figure(figsize=(16, 8))
    plot = sns.histplot(active_each_year_dataframe, legend=False, binwidth=binwidth, **kwargs);
    plot.set_ylabel('Count');
    plot.set_xlabel('Number of Active Technological Species');
    plot.xaxis.set_major_locator(MaxNLocator(integer=True));
    
    # https://stackoverflow.com/questions/31357611/format-y-axis-as-percent
    # could format axis to be percentages


def long_time(params, epoch_steps, final_year, P_tech_dominance = 0.9):
    n_epoch = final_year // epoch_steps

    index = 'year'
    columns = ['year', 'num_stars', 'num_planets', 'num_habitable_planets', 'num_life', 
           'num_complex_life', 'num_intelligent_life', 'num_cultural_life', 'N', 'N_extinct']

    future_of_life = pd.DataFrame(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
        columns = columns,
        index=[0]
    ).set_index(index)

    for i in range(n_epoch):
        temp_df = TimeDependentDrake(params, epoch_steps*(1+i), P_tech_dominance, df_input=future_of_life)
        future_of_life = future_of_life.append(temp_df.iloc[[-1]])

    return future_of_life


def weibull_mean(params):    
    return params['WEIBULL_SCALE_PARAMETER'] * gamma(1 + 1/params['WEIBULL_SHAPE_PARAMETER'])


def weibull_median(params):
    return params['WEIBULL_SCALE_PARAMETER'] * log(2) ** (1/params['WEIBULL_SHAPE_PARAMETER'])


def stars_within(how_many_lightyears):
    """
    based on data to nearest stars from www.atlasoftheuniverse.com
    curve fit in Excel (super fast and seems accurate)
    
    stars_within = [
        [4.2, 1.0],  # proxima centauri
        [12.5, 33],  # http://www.atlasoftheuniverse.com/12lys.html
        [20, 109],  # http://www.atlasoftheuniverse.com/20lys.html
        [250, 260_000],  # http://www.atlasoftheuniverse.com/250lys.html
        [5_000, 600_000_000],  # http://www.atlasoftheuniverse.com/5000lys.html
        [50_000, 200_000_000_000],  # http://www.atlasoftheuniverse.com/galaxy.html
    ]
    
    """
    return int(0.0319 * how_many_lightyears ** 2.7601)


def how_far(params, n_species):
    """
    based on data to nearest stars from www.atlasoftheuniverse.com
    curve fit in Excel (super fast and seems accurate)
    could use lookup to be slightly more accurate for closest stars
    """
        
    stars_per_lifeform = params['NUM_GALAXY'] / n_species
    
    # based on stars_within(), see Excel sheet and 
    avg_distance = (stars_per_lifeform / 0.0319) ** (1 / 2.7601)
    
    return round(avg_distance, 1)
