""" An implementation of the MinHash LSH method that operates on Dask DataFrames.

To do: Convert DataFrame operations to Array operations. This would eliminate
looping and increase computation speed.
"""

from random import random
import pandas as pd
import dask
import dask.dataframe as dd


class MinHashLSH():
    def __init__(self, df_a, df_b, id_col, hash_col, n_shingles, rows, rows_per_band, threshold, hash_func=hash):
        self.df_a = df_a[hash_col].to_frame()
        self.df_b = df_b[hash_col].to_frame()
        self.df_a_all_attributes = df_a
        self.df_b_all_attributes = df_b
        self.id_col = id_col
        self.hash_col = hash_col
        self.n_shingles = n_shingles
        self.rows_per_band = rows_per_band
        self.rows = rows
        self.threshold = threshold
        self.random_strings = [str(random()) for _ in range(self.rows)]
        self.hash_func = hash_func


    @staticmethod
    def shingles(value_to_hash, n_shingles):
        """Generate shingles of length n_shingles"""

        return {value_to_hash[i:i + n_shingles] for i in range(len(value_to_hash) - n_shingles + 1)}


    @staticmethod
    def signature_matrix(shingles, random_strings, rows_per_band, hash_func):
        """Calculate the signature matrix given sets of shingles, also apply the
        band_matrix transformation"""

        hasher = lambda x, i: abs(hash_func(x + i))
        signature =  [min([hasher(x, i) for x in shingles]) for i in random_strings]

        return MinHashLSH.band_matrix(signature, rows_per_band, hash_func)


    @staticmethod
    def band_matrix(signature_matrix, rows_per_band, hash_func):
        """Given a signature matrix, calculate the band matrix"""

        hasher = lambda x: abs(hash_func(x))

        return [hasher(tuple(signature_matrix[i:i + rows_per_band])) for i in range(len(signature_matrix)) if i % rows_per_band == 0]


    @staticmethod
    def jaccard(set_1, set_2):
        """Calculate Jaccard similarity given two sets"""

        return len(set_1.intersection(set_2)) / len(set_1.union(set_2))


    @staticmethod
    def get_band_proba(jaccard_sim, n_rows, n_bands):
        """Calculate the probability that at least one band will
        match given two records with Jaccard similarity =
        jaccard_sim

        Arguments:
        jaccard_sim (float): the jaccard sim theshold of interst, ex. 0.5
        n_rows (int): the number of minhash signatures
        n_bands (int): the number of bands generated from the minhash signatures

        """

        rows_per_band =  n_rows / n_bands
        probability =  1 - (1-jaccard_sim**rows_per_band)**n_bands

        return (probability, rows_per_band)


    @staticmethod
    def get_theshold_bump(n_rows, n_bands):
        """Calcualate the jaccard similarity theshold at which the
        probability of two records sharing a band matrix values begins
        to rapidly increase

        Arguments:
        n_rows (int): the number of minhash signatures
        n_bands (int): the number of bands generated from the minhash signatures
        """

        rows_per_band =  n_rows / n_bands

        return (1 / n_bands) ** (1 / rows_per_band)


    def apply_minhash_lsh(self, df):
        """Given a Dataframe, generate shingles, signature matrix, and band_matrix
        """

        df_shingles = df[self.hash_col].apply(MinHashLSH.shingles, args=(self.n_shingles,), meta=object)

        df_bands = df_shingles.apply(MinHashLSH.signature_matrix, args=(self.random_strings,
                                                                        self.rows_per_band,
                                                                        self.hash_func,), meta=object)

        return df_bands


    def get_band_matrix(self):
        """Apply the MinhashLSH process steps
        """

        df_a_bands = self.apply_minhash_lsh(self.df_a)
        df_b_bands = self.apply_minhash_lsh(self.df_b)

        return (df_a_bands, df_b_bands)


    def get_band_index(self, df):
        """SQL-style explode that take a datafram of record ids, list of band values
        and generates a new DataFrame with one row per id value and individual band value.

        id   band
        123  [000, 111, 222]

        to:

        id   band
        123  000
        123  111
        123  222
        """

        def func(df):
            return df.apply(pd.Series, 1) \
                     .stack() \
                     .reset_index(level=1, drop=True) \
                     .to_frame()

        band_index =  df.map_partitions(func)
        band_index = band_index.rename(columns = {0: self.hash_col})
        band_index[self.id_col] = band_index.index

        return band_index


    def join_bands(self):

        id_a_col = self.id_col + '_a'
        id_b_col = self.id_col + '_b'

        # Calculate shingles, minhash signature matrix, and band matrix
        bands_a, bands_b = self.get_band_matrix()

        # Transpose the band matrix to one row per id and band matrix value
        bands_a = self.get_band_index(bands_a)

        bands_b = self.get_band_index(bands_b)

        # Join the two band indexes on band values. This will connect all records
        # that share at least one band value. Then dedupe so that only distinct
        # id pairs are retained.
        joined_bands = bands_a.merge(bands_b, how='inner', on=self.hash_col,  suffixes=('_a', '_b'))
        joined_bands = joined_bands[[id_a_col, id_b_col]].drop_duplicates()


        # Calculate the shingles
        df_a_shingles = self.df_a[self.hash_col].apply(MinHashLSH.shingles,
                                                       args=(self.n_shingles,),
                                                       meta=object)

        df_b_shingles = self.df_b[self.hash_col].apply(MinHashLSH.shingles,
                                                       args=(self.n_shingles,),
                                                       meta=object)


        # Some cleanup here; also specifically setting types to ensure joins work
        # properly. The types seem to change throughout some processing steps. Also
        # changing the index type does not seem to stick, so copying the index into
        # another column and setting type before joining.
        df_a_shingles = df_a_shingles.to_frame()
        df_a_shingles.columns = ['shingles_a']
        df_a_shingles[id_a_col] = df_a_shingles.index
        df_a_shingles[id_a_col].astype(object)

        df_b_shingles = df_b_shingles.to_frame()
        df_b_shingles.columns = ['shingles_b']
        df_b_shingles[id_b_col] = df_b_shingles.index
        df_b_shingles[id_b_col].astype(object)

        joined_bands = joined_bands.merge(df_a_shingles, on= id_a_col, how='inner')
        joined_bands = joined_bands.merge(df_b_shingles, on= id_b_col, how='inner')
        joined_bands.astype(object)


        # Calculate the Jaccard similarity of the shingle pairs, retain only those
        # with similarities above the threshold
        joined_bands['sim'] = joined_bands.apply(lambda row: self.jaccard(row.shingles_a, row.shingles_b), axis=1, meta=float)
        final_columns = [id_a_col, id_b_col, 'sim']

        joined_bands = joined_bands[joined_bands.sim >= self.threshold][final_columns]

        # Explicitly setting types here, these seems to change during some procesing
        # steps and then prevent the tables from joining. Also, change the type of the
        # index does not seem to stick, so I'm copying it into a new column.
        joined_bands[id_a_col].astype(object)
        joined_bands[id_b_col].astype(object)

        self.df_a_all_attributes[id_a_col] = self.df_a_all_attributes.index
        self.df_a_all_attributes[id_a_col].astype(object)

        self.df_b_all_attributes[id_b_col] = self.df_b_all_attributes.index
        self.df_b_all_attributes[id_b_col].astype(object)


        # Join the above threshold pairs to the full input datasets, captures all
        # attribute columns
        joined_bands = joined_bands.merge(self.df_a_all_attributes, on=id_a_col, how='left')

        joined_bands = joined_bands.merge(self.df_b_all_attributes, on=id_b_col, how='left', suffixes=('_a', '_b'))

        # Need to reset index after computation since each Dask partition has a separate index
        computed_bands = joined_bands.compute()
        computed_bands.reset_index(inplace=True, drop=True)

        return computed_bands
