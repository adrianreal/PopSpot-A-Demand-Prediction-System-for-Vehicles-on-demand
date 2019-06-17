import pandas as pd
import Geohash as geo
import sys
import numpy as np
import argparse
from sets import Set
from math import radians

### The following two functions covert a timestamp to its corresponding 15-min period (0-95).
def to_period(time_stamp):
	hour, minute = time_stamp.split(':')
	periods_in_hour = int(hour) * 4
	periods_in_min = int(minute)/15
	cur_period = periods_in_hour + periods_in_min
	return cur_period

def periodize(data_file):
	data_file['timestamp'] = data_file['timestamp'].apply(to_period)
	return data_file

### This function calculates the great circle distance between any two points, but is not used in the program.
def great_circle_distance(lat1,lat2,lng1,lng2):
	deltaLat = lat2-lat1
	deltaLng = lng2-lng1
	a = sin(deltaLat/2.0)*sin(deltaLat/2.0)+cos(lat1)*cos(lat2)*sin(deltaLng/2.0)*sin(deltaLng/2.0)
	c = 2*arctan2(sqrt(a),sqrt(1-a))
	distDiff = 6371*c

### This function designs the historical short-term and long-term information and engineers new features for the training or testing dataset.
# Input: training/testing file.
# Output: training/testing file with additional engineered features, sorted in ascending order of (day, period).
def feature_gen(data_file, file_type):
	data_sort = data_file.sort_values(by=['day','timestamp'])
	min_day = min(data_file['day'].unique())
	if file_type == 'testing':
		max_day = max(data_file['day'].unique())
		max_day_period = max(data_file[data_file['day']==max_day]['timestamp'].unique())

	no_tuples, no_dimensions = data_file.shape
	prev_period = -1
	period = -1
	fine_dd = {}
	coarse_dd_B = {}
	coarse_dd_M = {}
	coarse_dd_S = {}
	period_list = []
	temp_dd = {}
	recorded_points = Set()
	temp_coarse_B = {}
	temp_coarse_M = {}
	temp_coarse_S = {}
	idx = -1 #Manually increment index instead of using index present in the dataframe due to rearrangement of dataframe objects (and their corresponding indices) earlier during sorting.

	# Create a list of historical aggregated demands for every point and every region. If there is no record at a point/region during a period, we assign a 0 for that point/region.
	for row in data_sort.itertuples():
		idx += 1
		geohash = str(row[1])
		day = int(row[2])
		period = int(row[3])
		dd = float(row[4])
		region_B = to_region_B[geohash]
		region_M = to_region_M[geohash]
		region_S = to_region_S[geohash]
	
                #for part in sgParts.itertuples(): 
                #        poly = part.geometry 
                #        if p.within(poly): 
                                #print part.Name 
                #                region = part.Name

		if idx == 0:	
			temp_dd[geohash] = dd
			temp_coarse_B[region_B] = dd
			temp_coarse_M[region_M] = dd
			temp_coarse_S[region_S] = dd

		# When the period of the current record is different from the period of the previous record, we append all the recorded demands of the previous period to the historical demand lists, while adding 0 to the historical demand lists that belong to points/regions that do not have any records during the previous period
		elif prev_period != period:
			if period-prev_period != 1 and not (period == 0 and prev_period == 95):
				empty_periods = period-prev_period-1
			else:
				empty_periods = 0
			period_list.append(prev_period)
			for ghash in all_geo_hashes:
				if ghash in temp_dd:
					if ghash in fine_dd:
						fine_dd[ghash].append(temp_dd[ghash])
					else: 
						fine_dd[ghash] = [temp_dd[ghash]]
				if ghash not in temp_dd:
					if ghash in fine_dd:
						fine_dd[ghash].append(0.0)
					else:
						fine_dd[ghash] = [0.0]
				
			for reg in all_regions_B:
				if reg in temp_coarse_B:
					if reg in coarse_dd_B:
						coarse_dd_B[reg].append(temp_coarse_B[reg])	
					else:
						coarse_dd_B[reg] = [temp_coarse_B[reg]]
				else:
					if reg in coarse_dd_B:
						coarse_dd_B[reg].append(0)
					else:
						coarse_dd_B[reg] = [0]

			for reg in all_regions_M:
				if reg in temp_coarse_M:
					if reg in coarse_dd_M:
						coarse_dd_M[reg].append(temp_coarse_M[reg])	
					else:
						coarse_dd_M[reg] = [temp_coarse_M[reg]]
				else:
					if reg in coarse_dd_M:
						coarse_dd_M[reg].append(0)
					else:
						coarse_dd_M[reg] = [0]

			for reg in all_regions_S:
				if reg in temp_coarse_S:
					if reg in coarse_dd_S:
						coarse_dd_S[reg].append(temp_coarse_S[reg])	
					else:
						coarse_dd_S[reg] = [temp_coarse_S[reg]]
				if reg not in temp_coarse_S:
					if reg in coarse_dd_S:
						coarse_dd_S[reg].append(0)
					else:
						coarse_dd_S[reg] = [0]
			
			# There are two instances where there are consecutive periods without any records at all, across all points and regions. Here we fill in 0 for these periods. 
			if empty_periods > 0:
				for i in range(empty_periods):
					for ghash in all_geo_hashes:
						fine_dd[ghash].append(0)
					for reg in coarse_dd_B:
						coarse_dd_B[reg].append(0)
					for reg in coarse_dd_M:
						coarse_dd_M[reg].append(0)
					for reg in coarse_dd_S:
						coarse_dd_S[reg].append(0)
					period_list.append(prev_period+i)
			
			temp_dd = {}
			temp_dd[geohash] = dd
			temp_coarse_B = {}
			temp_coarse_B[region_B] = dd
			temp_coarse_M = {}
			temp_coarse_M[region_M] = dd
			temp_coarse_S = {}
			temp_coarse_S[region_S] = dd

		# When the program reaches the end of the dataset, append all the recorded demands to the historical demand lists, while adding 0 to the historical demand lists that belong to points/regions that do not have any records during the this period

		elif idx == no_tuples-1:
			period_list.append(prev_period)
			temp_dd[geohash] = dd
			if region_B in temp_coarse_B:
				temp_coarse_B[region_B] += dd
			else:
				temp_coarse_B[region_B] = dd

			if region_M in temp_coarse_M:
				temp_coarse_M[region_M] += dd
			else:
				temp_coarse_M[region_M] = dd

			if region_S in temp_coarse_S:
				temp_coarse_S[region_S] += dd
			else:
				temp_coarse_S[region_S] = dd
			for ghash in all_geo_hashes:
				if ghash in temp_dd:
					if ghash in fine_dd:
						fine_dd[ghash].append(temp_dd[ghash])
					else: 
						fine_dd[ghash] = [temp_dd[ghash]]
				if ghash not in temp_dd:
					if ghash in fine_dd:
						fine_dd[ghash].append(0)
					else:
						fine_dd[ghash] = [0]
			
			for reg in all_regions_B:
				if reg in temp_coarse_B:
					if reg in coarse_dd_B:
						coarse_dd_B[reg].append(temp_coarse_B[reg])
					else:
						coarse_dd_B[reg] = [temp_coarse_B[reg]]
				else:
					if reg in coarse_dd_B:
						coarse_dd_B[reg].append(0)
					else:
						coarse_dd_B[reg] = [0]
			for reg in all_regions_M:
				if reg in temp_coarse_M:
					if reg in coarse_dd_M:
						coarse_dd_M[reg].append(temp_coarse_M[reg])	
					else:
						coarse_dd_M[reg] = [temp_coarse_M[reg]]
				else:
					if reg in coarse_dd_M:
						coarse_dd_M[reg].append(0)
					else:
						coarse_dd_M[reg] = [0]

			for reg in all_regions_S:
				if reg in temp_coarse_S:
					if reg in coarse_dd_S:
						coarse_dd_S[reg].append(temp_coarse_S[reg])	
					else:
						coarse_dd_S[reg] = [temp_coarse_S[reg]]
				else:
					if reg in coarse_dd_S:
						coarse_dd_S[reg].append(0)
					else:
						coarse_dd_S[reg] = [0]
	
		else:
			temp_dd[geohash] = dd
			if region_B in temp_coarse_B:
				temp_coarse_B[region_B] += dd
			else:
				temp_coarse_B[region_B] = dd

			if region_M in temp_coarse_M:
				temp_coarse_M[region_M] += dd
			else:
				temp_coarse_M[region_M] = dd

			if region_S in temp_coarse_S:
				temp_coarse_S[region_S] += dd
			else:
				temp_coarse_S[region_S] = dd
		prev_period = period
		prev_day = day
		prev_row = row
	
	## Feature Generation. Based on the historical information built earlier, this section generates additional historical features for each training sample
	# Generate additional historical/attribute features w.r.t. each tuple. Total features include: A) Attributional Features: 1. geohash ID, 2. Region(Small), 3. Region(Medium), 4. Region(Big), 5. Day-of-week, 6. Period. B) Short-term Historical Features: 1. Demand at this point (that this tuple corresponds to) over each of past 6 periods (6 features here), 2. Sum of demand at this point over past 2,4,6 periods (3 features here). C) Long-term Historical Features: Demand at this point during the current period over past 1,2 weeks and their average (3 features here). In total, considering point-granularity, there are 12 historical (both short and long-term) features. Repeat this for Region(Small), Region(Medium), Region(Big).

	# Altogether there are 54 features + 1 target variable 
	columns_count = 55
	# Build an empty matrix for filling in of feature tuples generated later. While there are 54 features, an extra empty column is created to include the target variable (the thing we want to predict), for conciseness. This target variable column will be separated later during training/testing.
	engineered_data = np.zeros([no_tuples,columns_count])
	data_idx = 0
	idx = -1
	# If the data is testing data, get the maximum number of periods available in the testing dataset. This is useful later to determine when does historical data stop, i.e. 5 periods before dataset end.
	if file_type == 'testing':
		total_periods = (max_day-min_day) * 96 + max_day_period + 1
	# For each record in the dataset, append the engineered features.
	for row in data_sort.itertuples():
		idx += 1
		feature_list = []
		day = int(row[2])
		day_of_week = day%7
		period = int(row[3])
		if day <= min_day+13: continue
		#if day == 61 and period >= 91: continue
		geohash = str(row[1])
		hash_id = to_id[geohash]
		(lat,lng) = geo.decode(geohash)
		lat = float(lat)
		lng = float(lng)
		dd = float(row[4])
		region_B = int(to_region_B[geohash])
		region_M = int(to_region_M[geohash])
		region_S = int(to_region_S[geohash])
		idx_in_list = (day-min_day) * 96 + period
		# Extract temporary demand sub-lists from historical demand lists for generation of short-term historical features.
		#dd = fine_dd[geohash][idx_in_list]
		dd_list = fine_dd[geohash][idx_in_list-6:idx_in_list]
		dd_S_list = coarse_dd_S[region_S][idx_in_list-6:idx_in_list]
		dd_M_list = coarse_dd_M[region_M][idx_in_list-6:idx_in_list]
		dd_B_list = coarse_dd_B[region_B][idx_in_list-6:idx_in_list]

		# If dataset is testing dataset, certain short-term historical demands may not be available for all points/regions. E.g. If the current record is recorded at T+5 (We are only allowed to generate features up to T), the only short-term historical demands (over past six periods) available are during T-1 and T. In these cases, extrapolation is done by filling these missing demands with its closest available recorded demand. In the above example, T2-T4 are filled with demand at T.
		if file_type == 'testing':
			if idx_in_list+1 > total_periods-4:
				periods_diff = idx_in_list+1-total_periods+4
				for i in range(periods_diff):
					j = 6-4+i
					dd_list[j] = dd_list[6-periods_diff-1]
					dd_S_list[j] = dd_S_list[6-periods_diff-1]
					dd_M_list[j] = dd_M_list[6-periods_diff-1]
					dd_B_list[j] = dd_B_list[6-periods_diff-1]

			dd_1 = dd_list[5]
			dd_2 = dd_list[4]
			dd_3 = dd_list[3]
			dd_4 = dd_list[2]
			dd_5 = dd_list[1]
			dd_6 = dd_list[0]
			dd_S_1 = dd_S_list[5]
			dd_S_2 = dd_S_list[4]
			dd_S_3 = dd_S_list[3]
			dd_S_4 = dd_S_list[2]
			dd_S_5 = dd_S_list[1]
			dd_S_6 = dd_S_list[0]
			dd_M_1 = dd_M_list[5]
			dd_M_2 = dd_M_list[4]
			dd_M_3 = dd_M_list[3]
			dd_M_4 = dd_M_list[2]
			dd_M_5 = dd_M_list[1]
			dd_M_6 = dd_M_list[0]
			dd_B_1 = dd_B_list[5]
			dd_B_2 = dd_B_list[4]
			dd_B_3 = dd_B_list[3]
			dd_B_4 = dd_B_list[2]
			dd_B_5 = dd_B_list[1]
			dd_B_6 = dd_B_list[0]

		else:
			dd_1 = fine_dd[geohash][idx_in_list-1]
			dd_2 = fine_dd[geohash][idx_in_list-2]
			dd_3 = fine_dd[geohash][idx_in_list-3]
			dd_4 = fine_dd[geohash][idx_in_list-4]
			dd_5 = fine_dd[geohash][idx_in_list-5]
			dd_6 = fine_dd[geohash][idx_in_list-6]

			dd_S_1 = coarse_dd_S[region_S][idx_in_list-1]
			dd_S_2 = coarse_dd_S[region_S][idx_in_list-2]
			dd_S_3 = coarse_dd_S[region_S][idx_in_list-3]
			dd_S_4 = coarse_dd_S[region_S][idx_in_list-4]
			dd_S_5 = coarse_dd_S[region_S][idx_in_list-5]
			dd_S_6 = coarse_dd_S[region_S][idx_in_list-6]

			dd_M_1 = coarse_dd_M[region_M][idx_in_list-1]
			dd_M_2 = coarse_dd_M[region_M][idx_in_list-2]
			dd_M_3 = coarse_dd_M[region_M][idx_in_list-3]
			dd_M_4 = coarse_dd_M[region_M][idx_in_list-4]
			dd_M_5 = coarse_dd_M[region_M][idx_in_list-5]
			dd_M_6 = coarse_dd_M[region_M][idx_in_list-6]

			dd_B_1 = coarse_dd_B[region_B][idx_in_list-1]
			dd_B_2 = coarse_dd_B[region_B][idx_in_list-2]
			dd_B_3 = coarse_dd_B[region_B][idx_in_list-3]
			dd_B_4 = coarse_dd_B[region_B][idx_in_list-4]
			dd_B_5 = coarse_dd_B[region_B][idx_in_list-5]
			dd_B_6 = coarse_dd_B[region_B][idx_in_list-6]

		sum6 = dd_1+dd_2+dd_3+dd_4+dd_5+dd_6
		sum4 = dd_1+dd_2+dd_3+dd_4
		sum2 = dd_1+dd_2
		dd_2week = fine_dd[geohash][idx_in_list-2*(96*7)]
		dd_1week = fine_dd[geohash][idx_in_list-(96*7)]
		dd_avg = (dd_2week+dd_1week)/2.0

		sum_S_6 = dd_S_1+dd_S_2+dd_S_3+dd_S_4+dd_S_5+dd_S_6
		sum_S_4 = dd_S_1+dd_S_2+dd_S_3+dd_S_4
		sum_S_2 = dd_S_1+dd_S_2
		dd_S_2week = coarse_dd_S[region_S][idx_in_list-2*(96*7)]
		dd_S_1week = coarse_dd_S[region_S][idx_in_list-(96*7)]
		dd_S_avg = (dd_S_2week+dd_S_1week)/2.0

		sum_M_6 = dd_M_1+dd_M_2+dd_M_3+dd_M_4+dd_M_5+dd_M_6
		sum_M_4 = dd_M_1+dd_M_2+dd_M_3+dd_M_4
		sum_M_2 = dd_M_1+dd_M_2
		dd_M_2week = coarse_dd_M[region_M][idx_in_list-2*(96*7)]
		dd_M_1week = coarse_dd_M[region_M][idx_in_list-(96*7)]
		dd_M_avg = (dd_M_2week+dd_M_1week)/2.0

		sum_B_6 = dd_B_1+dd_B_2+dd_B_3+dd_B_4+dd_B_5+dd_B_6	
		sum_B_4 = dd_B_1+dd_B_2+dd_B_3+dd_B_4
		sum_B_2 = dd_B_1+dd_B_2
		dd_B_2week = coarse_dd_B[region_B][idx_in_list-2*(96*7)]
		dd_B_1week = coarse_dd_B[region_B][idx_in_list-(96*7)]
		dd_B_avg = (dd_B_2week+dd_B_1week)/2.0

		# Form a new tuple with these features and adding it to the engineered dataset.
		feature_list.append(dd)
		feature_list.append(int(hash_id))
		feature_list.append(int(region_S))
		feature_list.append(int(region_M))
		feature_list.append(int(region_B))
		feature_list.append(int(day_of_week))
		feature_list.append(int(period))
		feature_list.append(dd_1)
		feature_list.append(dd_2)
		feature_list.append(dd_3)
		feature_list.append(dd_4)
		feature_list.append(dd_5)
		feature_list.append(dd_6)
		feature_list.append(sum6)
		feature_list.append(sum4)
		feature_list.append(sum2)
		feature_list.append(dd_2week)
		feature_list.append(dd_1week)
		feature_list.append(dd_avg)
		feature_list.append(dd_S_1)
		feature_list.append(dd_S_2)
		feature_list.append(dd_S_3)
		feature_list.append(dd_S_4)
		feature_list.append(dd_S_5)
		feature_list.append(dd_S_6)
		feature_list.append(sum_S_6)
		feature_list.append(sum_S_4)
		feature_list.append(sum_S_2)
		feature_list.append(dd_S_2week)
		feature_list.append(dd_S_1week)
		feature_list.append(dd_S_avg)
		feature_list.append(dd_M_1)
		feature_list.append(dd_M_2)
		feature_list.append(dd_M_3)
		feature_list.append(dd_M_4)
		feature_list.append(dd_M_5)
		feature_list.append(dd_M_6)
		feature_list.append(sum_M_6)
		feature_list.append(sum_M_4)
		feature_list.append(sum_M_2)
		feature_list.append(dd_M_2week)
		feature_list.append(dd_M_1week)
		feature_list.append(dd_M_avg)
		feature_list.append(dd_B_1)
		feature_list.append(dd_B_2)
		feature_list.append(dd_B_3)
		feature_list.append(dd_B_4)
		feature_list.append(dd_B_5)
		feature_list.append(dd_B_6)
		feature_list.append(sum_B_6)
		feature_list.append(sum_B_4)
		feature_list.append(sum_B_2)
		feature_list.append(dd_B_2week)
		feature_list.append(dd_B_1week)
		feature_list.append(dd_B_avg)
		
		engineered_data[data_idx,:] = feature_list
		data_idx += 1
		sys.stdout.write('\r Progress {:.2f}%'.format((idx+1) * 100.0 / no_tuples))
		sys.stdout.flush()
	print '\n'
	print 'Saving File \n'
	engineered_data.resize((data_idx,columns_count))
	engineered_data = pd.DataFrame(engineered_data)
	return engineered_data

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_raw', type=str)
	parser.add_argument('--test_raw', type=str)
	args = parser.parse_args()
	train_path = args.train_raw
	test_path = args.test_raw
	test_path = args.test_raw
	train_data = pd.read_csv(train_path, skiprows = 0)
	test_data = pd.read_csv(test_path, skiprows = 0)
	train_data = periodize(train_data)
	test_data = periodize(test_data)

	# Assign a unique ID for each geohash6.
	all_geo_hashes = list(Set(train_data["geohash6"].unique().tolist() + test_data["geohash6"].unique().tolist()))
	to_id = {}
	hash_id = 0
	for ghash in all_geo_hashes:
		to_id[ghash] = hash_id
		hash_id += 1
	"""
	near_neighbours = {}
	for source_hash in all_geo_hashes:
		(source_lat,source_lng) = geo.decode(str(source_hash))
		source_lat = float(source_lat)
		souce_lng = float(source_lng)
		near_neighbours[source_hash] = [source_hash]
		for neighbour_hash in all_geo_hashes:
			print neighbour_hash
			print source_hash
			if str(neighbour_hash) == str(source_hash): continue
			(n_lat,n_lng) = geo.decode(str(neighbour_hash))
			neighbour_lat = float(n_lat)
			
			neighbour_lng = float(n_lng)
			print (source_lat,source_lng,neighbour_lat,neighbour_lng)
			source_lat,neighbour_lat,source_lng,neighbour_lng = map(radians,[source_lat,neighbour_lat,source_lng,neighbour_lng])
			dist_apart = great_circle_distance(source_lat,neighbour_lat,source_lng,neighbour_lng)
			if dist_apart <= 0.2:
				near_neighbours[source_hash].append(neighbour_hash)
	print "Neighbours"
	for hash_list in near_neighbours:
		print len(hash_list)
	"""

	# Split map into 12x12, 8x8, 5x5 grids of region(Small), region(Medium), and region(Big) respectively. For each geohash, assign a region(Small), region(Medium), and region(Big) to it.
	# Information of the boundaries of all recorded points are available based on prior analysis.
	min_lat = -5.48
	max_lat = -5.24
	min_lng = 90.6
	max_lng = 91.0

	n_intervals_S = 5
	lat_size_S = (max_lat-min_lat)/n_intervals_S
	lng_size_S = (max_lng-min_lng)/n_intervals_S
	n_intervals_M = 8
	lat_size_M = (max_lat-min_lat)/n_intervals_M
	lng_size_M = (max_lng-min_lng)/n_intervals_M
	n_intervals_B = 12
	lat_size_B = (max_lat-min_lat)/n_intervals_B
	lng_size_B = (max_lng-min_lng)/n_intervals_B
	all_regions_S = []
	all_regions_M = []
	all_regions_B = []
	to_region_S = {}
	to_region_M = {}
	to_region_B = {}

	for i in range(n_intervals_S):
		for j in range(n_intervals_S):
			temp_region = i * n_intervals_S + j
			all_regions_S.append(temp_region)
	for i in range(n_intervals_M):
		for j in range(n_intervals_M):
			temp_region = i * n_intervals_M + j
			all_regions_M.append(temp_region)
	for i in range(n_intervals_B):
		for j in range(n_intervals_B):
			temp_region = i * n_intervals_B + j
			all_regions_B.append(temp_region)
	for ghash in all_geo_hashes:
		gps = geo.decode(ghash)
		lat, lng = gps
		lat = float(lat)
		lng = float(lng)
		
		lat_region_B = int((lat-min_lat)/lat_size_B)
		if lat_region_B < 0: lat_region_B = 0
		elif lat_region_B >= n_intervals_B: lat_region_B = n_intervals_B-1
		lng_region_B = int((lng-min_lng)/lng_size_B)
		if lng_region_B < 0: lng_region_B = 0
		elif lng_region_B >= n_intervals_B: lng_region_B = n_intervals_B-1
		region_B = lat_region_B * n_intervals_B + lng_region_B
		to_region_B[ghash] = region_B

		lat_region_M = int((lat-min_lat)/lat_size_M)
		if lat_region_M < 0: lat_region_M = 0
		elif lat_region_M >= n_intervals_M: lat_region_M = n_intervals_M-1
		lng_region_M = int((lng-min_lng)/lng_size_M)
		if lng_region_M < 0: lng_region_M = 0
		elif lng_region_M >= n_intervals_M: lng_region_M = n_intervals_M-1
		region_M = lat_region_M * n_intervals_M + lng_region_M
		to_region_M[ghash] = region_M

		lat_region_S = int((lat-min_lat)/lat_size_S)
		if lat_region_S < 0: lat_region_S = 0
		elif lat_region_S >= n_intervals_S: lat_region_S = n_intervals_S-1
		lng_region_S = int((lng-min_lng)/lng_size_S)
		if lng_region_S < 0: lng_region_S = 0
		elif lng_region_S >= n_intervals_S: lng_region_S = n_intervals_S-1
		region_S = lat_region_S * n_intervals_S + lng_region_S
		to_region_S[ghash] = region_S

	## Design historical short-term and long-term information and feature engineering for both training and testing data.
	print "Feature Engineering for Training Data"
	train_data = feature_gen(train_data,'train')
	train_data.to_csv("data/engineered_train.csv", index = False)
	print "Feature Engineering for Testing Data"
	test_data = feature_gen(test_data,'testing')
	test_data.to_csv("data/engineered_test.csv", index = False)
