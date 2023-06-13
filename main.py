import IsoPhotRad
import os
import PreSN_LC


# Text file should include path to dir with bunch of Fitz images of one SN
# It will then loop through all these and make back aps for all
# The name of bkg apps files will be from fitz file name
# Fitz file name comp:
# SN_Name.band.data_msc
# Pull name and date from there


# The input txt file should be set up as such:
# SN NAME
# Absolute file path for the fitz file directory
# RA
# Dec
# radius of aperture
# time of explosion
# absolute file path for photometry files
# absolute file path for diffim stats files
# flag for which script to run: 0 for Light Curve and 1 for IsoPhotRad

print(os.getcwd())

path = input("ENTER TXT FILE ABSOLUTE PATH.\nTHE TXT FILE SHOULD CONTAIN ABSOLUTE PATHS - ")

# load in the input text file and split up the contents
txt_file = open(path, 'r')
txt = txt_file.read()
txt_arr = txt.split('\n')

# to determine what the next step is we look at the ninth entry of the text file
# 0 - we run the LC script
# 1 - we run the IsoPhotRad  script

flag = float(txt_arr[8]) == 0

if not flag:
    sn_name = txt_arr[0]
    fitz_img_dir = txt_arr[1]
    ra = float(txt_arr[2])
    dec = float(txt_arr[3])
    r_ap = float(txt_arr[4])

    # now we go into that directory and make an array storing all of the fitz file names
    fitz_img_lst = os.listdir(fitz_img_dir)

    # this will be where all the bkg apps will be saved. 
    # we check to see if it already exists, making it if it doesnt
    bkg_apps_dir_path = os.path.join(os.getcwd(), "bkg_apps")
    if not os.path.exists(bkg_apps_dir_path):
        os.mkdir(bkg_apps_dir_path)

    for f in fitz_img_lst:
        # we make sure its a fits file
        if not f.endswith(".fits"):
            continue

        # list only contains the name, so we combine with current directory to get full path
        fitz_img_path = os.path.join(fitz_img_dir, f)

        # lets decompose the fitz file to name our output file
        # example is '2020tlf.g.200912_2281713_2199.069.sw.fits'
        fitz_name_decomp = f.split('.')
        sn_name = fitz_name_decomp[0]
        band_filter = fitz_name_decomp[1]
        fitz_name_decomp = fitz_name_decomp[2].split('_')
        sn_date = fitz_name_decomp[0]
        bkg_apps_dir_path_withfitsname = os.path.join(bkg_apps_dir_path, sn_name)
        
        # we check to see if it already exists, making it if it doesnt
        if not os.path.exists(bkg_apps_dir_path_withfitsname):
            os.mkdir(bkg_apps_dir_path_withfitsname)
        
        bkg_apps_dir_path_withfitsname = os.path.join(bkg_apps_dir_path_withfitsname, band_filter)
        if not os.path.exists(bkg_apps_dir_path_withfitsname):
            os.mkdir(bkg_apps_dir_path_withfitsname)

        bkg_apps_dir_path_withfitsname = bkg_apps_dir_path_withfitsname

        # now call the primary function that will calculate and save the bkg appertures
        IsoPhotRad.main(fitz_img_path, ra, dec, r_ap, bkg_apps_dir_path_withfitsname, sn_date)
else:
    sn_name = txt_arr[0]
    t_explosion = float(txt_arr[5])
    photometry_path = txt_arr[6]
    diffim_path = txt_arr[7]

    PreSN_LC.main(sn_name, t_explosion, photometry_path, diffim_path)


# path = input("ENTER SN TXT FILE DIRECTORY NAME: ")
# path = os.path.join(os.getcwd(), path)
# file_lst = os.listdir(path)
# dir_path = os.path.join(path, "bkg_apps")
# if not os.path.exists(dir_path):
#     os.mkdir(dir_path)

# for f in file_lst:
#     # you have to combine path with current director
#     file_path = os.path.join(path, f)
#     if not file_path.endswith(".txt"):
#         continue
#     print(file_path)
#     IsoPhotRad.main(file_path, dir_path)