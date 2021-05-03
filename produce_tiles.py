from os import path, makedirs
# from datetime import datetime, timedelta
import pandas as pd
# from dateutil.rrule import rrule, MINUTELY
from split_image import split_image

SIDE_LEN = 512
path_all_csv = 'data_csv/path_all.csv'

def apply_row(row, data_store_path):
    band1_store_path = path.join(data_store_path, 'band1', row.timestamp.strftime("%Y-%m-%d"))
    makedirs(band1_store_path, exist_ok=True)
    band3_store_path = path.join(data_store_path, 'band3', row.timestamp.strftime("%Y-%m-%d"))
    makedirs(band3_store_path, exist_ok=True)
    hms_store_path = path.join(data_store_path, 'hms', row.timestamp.strftime("%Y-%m-%d"))
    makedirs(hms_store_path, exist_ok=True)
    split_image(
        side_length=SIDE_LEN,
        band1_path=row.path_band1,
        band1_store_path=band1_store_path,
        band3_path=row.path_band3,
        band3_store_path=band3_store_path,
        mask_path=row.path_hms,
        mask_store_path=hms_store_path,
        daynight_path=row.path_daynight,
        basename=row.timestamp.strftime("%Y-%m-%d-%H%M")
        )
    return 1


if __name__ == "__main__":
    # specify data storage path and create directory
    data_store_dir = str(SIDE_LEN) + "x" + str(SIDE_LEN)
    data_store_path = path.join("/n/holyscratch01/mickley/hms_vision_data/", data_store_dir)
    makedirs(data_store_path, exist_ok=True)

    # load csv as dataframe
    path_all_df = pd.read_csv(path_all_csv, index_col='timestamp', parse_dates=['timestamp'])

    # create progress column if necessary
    progress_col = 'split_' + str(SIDE_LEN)
    if not progress_col in path_all_df:
        path_all_df[progress_col] = 0
    timestamp_list = path_all_df.sort_index().index
    path_all_df.to_csv(path_all_csv)

    for timestamp in timestamp_list:
        row = path_all_df.loc[timestamp, :]
        if row[progress_col] == 1:
            continue
        else:
            band1_store_path = path.join(data_store_path, 'band1', timestamp.strftime("%Y-%m-%d"))
            makedirs(band1_store_path, exist_ok=True)
            band3_store_path = path.join(data_store_path, 'band3', timestamp.strftime("%Y-%m-%d"))
            makedirs(band3_store_path, exist_ok=True)
            hms_store_path = path.join(data_store_path, 'hms', timestamp.strftime("%Y-%m-%d"))
            makedirs(hms_store_path, exist_ok=True)
            split_image(
                side_length=SIDE_LEN,
                band1_path=row.path_band1,
                band1_store_path=band1_store_path,
                band3_path=row.path_band3,
                band3_store_path=band3_store_path,
                mask_path=row.path_hms,
                mask_store_path=hms_store_path,
                daynight_path=row.path_daynight,
                basename=timestamp.strftime("%Y-%m-%d-%H%M")
                )
            row[progress_col] = 1
            path_all_df.to_csv(path_all_csv)
            print(timestamp, "done")


    # path_all_df.apply(lambda row: apply_row(row, data_store_path), axis='columns')


    path_all_df.to_csv(path_all_csv)

    # iterate over the time range every 30 min
    # prev_date = None
    # for dt in rrule(freq=MINUTELY, interval=30, dtstart=start_time, until=until_time):
    #     band1 = band1_df.loc[band1_df['timestamp'] == dt]
    #     band3 = band3_df.loc[band3_df['timestamp'] == dt]
    #     hms = hms_df.loc[hms_df['timestamp'] == dt]
    #     any_empty = any([band1.empty, band3.empty, hms.empty])
    #     if any_empty:
    #         continue
    #     else:
    #         done_col = 'done_{len}'.format(len=str(SIDE_LEN))
    #         if band1[done_col].values[0] and band3[done_col].values[0] and hms[done_col].values[0]:
    #             continue
    #         else:
    #             if prev_date != dt.date():
    #                 band1_store_path = path.join(data_path, 'band1', dt.strftime("%Y-%m-%d"))
    #                 makedirs(band1_store_path, exist_ok=True)
    #                 band3_store_path = path.join(data_path, 'band3', dt.strftime("%Y-%m-%d"))
    #                 makedirs(band3_store_path, exist_ok=True)
    #                 hms_store_path = path.join(data_path, 'hms', dt.strftime("%Y-%m-%d"))
    #                 makedirs(hms_store_path, exist_ok=True)
    #             prev_date = dt.date()

    #             start_of_year = dt.replace(month=1, day=1)
    #             yday = (dt - start_of_year + timedelta(days=1)).days
    #             td = timedelta(hours=dt.hour, minutes=dt.minute)
    #             daynight = daynight_df.loc[(daynight_df["yday"] == yday) & (daynight_df["time"] == td)]

    #             split_image(
    #                 side_length=SIDE_LEN,
    #                 band1_path=band1['path'].values[0],
    #                 band1_store_path=band1_store_path,
    #                 band3_path=band3['path'].values[0],
    #                 band3_store_path=band3_store_path,
    #                 mask_path=hms['path'].values[0],
    #                 mask_store_path=hms_store_path,
    #                 daynight_path=daynight['path'].values[0],
    #                 basename=dt.strftime("%Y-%m-%d-%H%M")
    #                 )

    #             band1_df.at[band1_df['timestamp'] == dt, done_col] = 1
    #             band1_df.to_csv('band1.csv', index=False)
    #             band3_df.at[band3_df['timestamp'] == dt, done_col] = 1
    #             band3_df.to_csv('band3.csv', index=False)
    #             hms_df.at[hms_df['timestamp'] == dt, done_col] = 1
    #             hms_df.to_csv('HMS.csv', index=False)
    #             print("Finished tiling {band1}, {band3}, and {mask}!".format(band1=band1['path'].values[0],
    #                                                                          band3=band3['path'].values[0],
    #                                                                          mask=hms['path'].values[0]))
    # print("All done!")
