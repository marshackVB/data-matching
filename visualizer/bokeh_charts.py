# Generate Bokeh charts
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure


class BinnedValuesChart():
    def __init__(self, df, id_col, bin_col, n_bins, title, plot_size_and_tools):
        self.df = df[[id_col, bin_col]]
        self.id_col = id_col
        self.bin_col = bin_col
        self.n_bins = n_bins
        self.title = title
        self.plot_size_and_tools = plot_size_and_tools


    @staticmethod
    def get_prob_bins(x, n):
        return int((x * 100) // n)


    def get_binned_values(self, df, id_col, bin_col, n_bins):

        df['bin'] = df[bin_col].apply(lambda x: BinnedValuesChart.get_prob_bins(x, n_bins))
        df = df.groupby('bin')[id_col].count()
        df = df.to_frame()
        df.columns = ['counts']
        df.reset_index(inplace=True)
        df['total'] = df.counts.sum()
        df['percent'] = df.apply(lambda row: int(row.counts/row.total * 100), axis=1)

        return df


    def get_binned_chart(self, dataframe, title):

        source = ColumnDataSource(dataframe)

        binned_chart = figure(title=title, **self.plot_size_and_tools)

        binned_chart.vbar(x='bin', top='percent', width=0.9, alpha=0.8, source=source)

        hover = binned_chart.select(dict(type=HoverTool))
        hover.tooltips = [('Percent', '@percent{00}')]

        return binned_chart


    def generate_binned_chart(self):
        binned_values = self.get_binned_values(self.df, self.id_col, self.bin_col, self.n_bins)
        binned_chart = self.get_binned_chart(binned_values, self.title)
        return binned_chart
