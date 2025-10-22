import h5py

with h5py.File("/mnt/data0/xiaoxiong/atm_libero/libero_spatial_2/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo.hdf5") as F:
    import ipdb; ipdb.set_trace()
    print(F.keys())
    