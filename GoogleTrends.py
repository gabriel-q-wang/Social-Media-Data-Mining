from pytrends.request import TrendReq
import pandas as pd


def main():
    # Set up api wrapper
    pytrends = TrendReq(hl='en-US', tz=360)

    # Limit of 5 keywords
    kw_list = ["Steelcase"]

    # Build pipeline
    pytrends.build_payload(kw_list, cat=0, timeframe='all', geo='', gprop='')

    # Get overall interest over the entire timeline
    interestDF = pytrends.interest_over_time()
    interestDF.to_csv(
        "C:\\Users\\gwang\\Documents\\01 ADS Projects\\GoogleTrends5YearInterest_test.csv",
        index=True)
    print(interestDF.head())
    print()

    # Sleep 60 prevents you from being rate limited
    # Get hourly interest over the time set
    hourlyDF = pytrends.get_historical_interest(kw_list, year_start=2019, month_start=7, day_start=1, hour_start=0,
                                                year_end=2019, month_end=7, day_end=1, hour_end=1, cat=0, geo='',
                                                gprop='', sleep=60)
    hourlyDF.to_csv(
        "C:\\Users\\gwang\\Documents\\01 ADS Projects\\GoogleTrends5YearHourlyInterest_test.csv",
        index=True)
    print(hourlyDF.head())
    print()

    # Get regional interest across the world
    # Can switch to state or city specific
    regionDF = pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True, inc_geo_code=False)
    regionDF.to_csv(
        "C:\\Users\\gwang\\Documents\\01 ADS Projects\\GoogleTrendsRegionInterest_test.csv",
        index=True)
    print(regionDF.head())
    print()

    # Get rising related topics
    risingDF = pytrends.related_topics().get('Steelcase').get('rising')
    risingDF.to_csv(
        "C:\\Users\\gwang\\Documents\\01 ADS Projects\\GoogleTrendsRisingRelated_test.csv",
        index=True)
    # Get top related topics
    topDF = pytrends.related_topics().get('Steelcase').get('top')
    topDF.to_csv(
        "C:\\Users\\gwang\\Documents\\01 ADS Projects\\GoogleTrendsTopRelated_test.csv",
        index=True)
    print(risingDF.head())
    print()
    print(topDF.head())


if __name__ == "__main__":
    # calling main function
    main()
