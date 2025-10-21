import h5py 
with h5py.File("/home/ZhangChuye/Documents/vik_module/data/lb90_8tk_raw/LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate_demo.hdf5") as f:
    print(f['data']['demo_0']['obs'].keys())