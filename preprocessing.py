import pandas as pd

df = pd.read_csv('Harvard Fees Dataset.csv')
pivot_df = df.pivot(index='academic.year', columns='school', values='cost')
pivot_df.reset_index(inplace=True)

inflation_data = {
    'academic.year': list(range(1985, 2018)),
    'inflation_rate': [
        3.54,  # 1985
        1.91,  # 1986
        3.65,  # 1987
        4.08,  # 1988
        4.83,  # 1989
        5.39,  # 1990
        4.25,  # 1991
        3.03,  # 1992
        2.96,  # 1993
        2.61,  # 1994
        2.81,  # 1995
        2.93,  # 1996
        2.34,  # 1997
        1.55,  # 1998
        2.19,  # 1999
        3.38,  # 2000
        2.83,  # 2001
        1.59,  # 2002
        2.27,  # 2003
        2.68,  # 2004
        3.39,  # 2005
        3.24,  # 2006
        2.85,  # 2007
        3.84,  # 2008
        -0.36, # 2009
        1.64,  # 2010
        3.16,  # 2011
        2.07,  # 2012
        1.46,  # 2013
        1.62,  # 2014
        0.12,  # 2015
        1.26,  # 2016
        2.13   # 2017
    ]
}
inflation_df = pd.DataFrame(inflation_data)

processed_df = pivot_df.merge(inflation_df, on='academic.year', how='left')
processed_df.to_csv('dataset.csv', index=False)