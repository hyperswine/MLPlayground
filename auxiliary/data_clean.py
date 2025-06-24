"""
Script to clean GSMArena (2020) data. Unfortunately some 'boiler-plate' code still exists in the notebooks for now, though
more scripts may be written to streamline the process.
"""
import re
import pandas as pd
from functools import reduce


def launch_announced(string):
  """
  :return: First regexed year.
  """
  year = re.search(r"\d{4}", str(string))
  return year.group(0) if year else None


def available_discontinued(string):
  """
  :return: 1 if available, 0 if discontinued/cancelled.
  """
  return str(1) if "Available" in string else str(0)


def usb_type(string):
  """
  'comms_usb'
  :return: 3 if 3.1 type-C, 2 if type-C, 1 if proprietary, 0 if microusb or other
  """
  string = str(string)
  if ("31" in string or "30" in string) and "Type-C" in string:
    return str(3)
  if "Type-C" in string:
    return str(2)
  if "proprietary" in string.lower():
    return str(1)

  return str(0)


def squared_dimensions(string):
  """
  "body_dimensions"
  :return: All the mm dimensions squared
  """
  dimensions_m = string.split('m')
  # dimensions = dimensions_m[0].split('x')

  dimensions = [t for t in dimensions_m[0].split()
                if t.lstrip('+-').replace('.', '', 1).isdigit()]
  print(dimensions)

  try:
    return str(reduce(lambda r, d: float(r) * float(d), dimensions))
  except:
    print("failed for {}".format(string))
    return None


def extract_screen_in(string):
  """
  "display_size"
  :return: 10X inches of screen & 10X screen-to-body ratio%, e.g. (65, 845)
  """
  if not string:
    return string

  # sizes = [t for t in re.split(r'[~% ]', string)
  #          if t.lstrip('(+~-').rstrip('%').replace('.', '', 1).isdigit()]

  # try:
  #     return sizes[1], sizes[2]
  # except:
  #     try:
  #         print("no screen-to-body")
  #         return sizes[1]
  #     except:
  #         print("nothing found!")
  #         return None

  return string


def wlan_extract(string):
  """
  :return: 4 if Wifi-6 enabled, 3 if dual-band ac, 2 if Wifi-direct/hotspot, 1 if only Wifi 802.11/primitive
  networking, 0 if no wifi.
  """
  string = string.lower()

  if '6' in string and 'wifi' in string:
    return str(4)
  if 'ac' in string and 'wifi' in string:
    return str(3)
  if 'direct' in string or 'hotspot' in string:
    return str(2)
  if 'wifi' in string:
    return str(1)

  return str(0)


# NOTE: due to the differing orders of the sensor words, its not possible to simply use a labelencoder
def sensor_extract(string):
  """
  :return: 6 if accelerometer, proximity, compass, gyro, fingerprint, barometer
           5 if any 5 of them
           4 if any 4 of them
           3 if any 3 of them
           2 if 2
           1 if 1
           0 otherwise, i.e. no sensors
  """
  sensors = ['accelerometer', 'proximity',
             'compass', 'gyro', 'fingerprint', 'barometer']
  string = string.lower()

  count = 0
  for sensor in sensors:
    if sensor in string:
      count += 1

  return count


def os_extract(string):
  """
  :return: 3 if android, 2 if apple, 1 if microsoft, 0 if proprietary or no official OS
  """
  string = str(string)
  if "Android" in string:
    return str(3)
  if "iOS" in string:
    return str(2)
  if "Microsoft" in string:
    return str(1)

  return str(0)


def cpu_cores(string):
  """
  :return: 3 if octa-core, 2 if quad-core, 1 if dual-core, 0 if not specified/low-end hardware
  """
  cores = ["Dual", "Quad", "Octa"]
  for core in cores:
    if core in string:
      return str(cores.index(core) + 1)

  return str(0)


def cpu_clock(string):
  """
  'platform_cpu' -> cpu_clock
  NOTE: this takes in the same column as 'cpu_cores', i.e., platform_cpu
  :return: The clock speed of each individual core.
      e.g. = return '395' for 3.95GHz
  """
  string = str(string)
  cpu_core = cpu_cores(string)

  print(string)
  if 'GHz' in string:
    l = [t for t in re.split('GHz', string)]
  elif 'MHz' in string:
    l = [t for t in re.split('MHz', string)]
  else:
    print("idk")
    return None

  l2 = [t for t in re.split(r'[~%x/ ]', l[0])
        if t.lstrip('(+~-').replace('.', '', 1).isdigit()]
  clk_spd = l2[-1]
  if 'MHz' in string:
    clk_spd = float(clk_spd) / 1000

  return cpu_core, clk_spd


def gpu_platform_extract(string):
  """
  :return: 3 if Adreno, 2 if Mali, 1 if PowerVr, 0 if proprietary/ low-end.
  """
  string = str(string)

  cpu_arch = ["powervr", "mali", "adreno"]
  for cpu in cpu_arch:
    if cpu in string.lower():
      return str(cpu_arch.index(cpu) + 1)

  return 0


def cam_vid(string):
  """
  :return: The camera (selfie or main) in 'p' (2K, 4K converted to 'p')
  TODO: make this work
  """
  string = str(string)

  p_string = re.search(r"\d+p", string)
  k_string = ""

  if not p_string:
    k_string = re.search(r"\d+", string.lower())
    if not k_string:
      return string
    if '2' in k_string.group(0):
      return '1080p'
    if '4' in k_string.group(0):
      return '2160p'

  return string


def split_display(df):
  """
  :return:  splits display_size column into screen_size and screen_body_ratio
  """
  d = pd.DataFrame(df['display_size'].tolist(), index=df.index)

  if len(d) < 2:
    return df

  # Check all screen_body_ratio is less than 100%
  set(d[1].apply(lambda x: float(x) < 100))

  df['screen_size'], df['screen_body_ratio'] = d[0], d[1]
  del df['display_size']
  return df


def re_cpu_clock(df):
  """
  :return:  splits platform_cpu column into core_count and clk_speed
  """
  d = pd.DataFrame(df['platform_cpu'].tolist(), index=df.index)

  if len(d) < 2:
    return df

  df['core_count'], df['clk_speed'] = d[0], d[1]
  del df['platform_cpu']
  return df


# TODO 1 -> encode 'OEM' with label-encoder after lower().
# TODO 2 -> remove outliers, if they are 1.5X IQR.
def clean_data(df):
  """
  Run all the functions to clean data
  Take in a dataframe as input, and apply all functions to it.
  """

  # Drop rows with null values
  # NOTE: allowing NaN for now.
  # df.dropna(inplace=True, axis=0, how='any')
  # df.reset_index(drop=True)

  # Feature names
  straight_features = ["memory_internal",
                       "main_camera_single", "main_camera_video", "misc_price",
                       "selfie_camera_video",
                       "selfie_camera_single", "battery"]

  f_map = {"launch_announced": launch_announced, "launch_status": available_discontinued,
           "body_dimensions": squared_dimensions, "display_size": extract_screen_in, "comms_wlan": wlan_extract,
           "comms_usb": usb_type,
           "features_sensors": sensor_extract, "platform_os": os_extract, "platform_cpu": cpu_clock,
           "platform_gpu": gpu_platform_extract}

  # Map all straight features
  for feature in straight_features:
    s = df[feature]
    df[feature] = s.str.extract(r"(\d+)")

  # Map all other numeric & categorical features specifically
  for feature in f_map:
    f = f_map.get(feature)
    df[feature] = df[feature].apply(f)

  # Clean up & ensure right form.
  df.main_camera_single = df["main_camera_single"].apply(cam_vid)
  df = split_display(df)
  df = re_cpu_clock(df)

  # Convert everything to floating point & return result
  df_clean = df.infer_objects()
  print(df_clean.dtypes)
  return df_clean


if __name__ == '__main__':
  """
  Test the cleaning with the same datasets, before using them in actual .py Machine Learning Scripts.
  """

  print("opening dataset...")
  # Open Dataset
  data = pd.read_csv('dataset/GSMArena_dataset_2020.csv', index_col=0)

  data_features = data[
      ["launch_announced", "launch_status", "body_dimensions", "display_size", "comms_wlan", "comms_usb",
       "features_sensors", "platform_os", "platform_cpu", "platform_gpu", "memory_internal",
       "main_camera_single", "main_camera_video", "misc_price",
       "selfie_camera_video",
       "selfie_camera_single", "battery"]]

  print("cleaning data...")
  # Clean data
  dfX = clean_data(data_features)

  print("Some insights on original")
  # Gain some insights on the 'cleaned' data.
  print(dfX.info())
  print(dfX.head())
