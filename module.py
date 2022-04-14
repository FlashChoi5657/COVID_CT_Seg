import numpy as np
import scipy.ndimage
from skimage import measure
from skimage.morphology import convex_hull_image
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
import SimpleITK as sitk


def Resize_128(data, img_info):
    # img_info[1] Direction matrix. Default is identity. mapping, rotation
    # img_info[2] Origin coordinates of the voxel with index (0,0,0) in physical units
    # img_info[3] Distance between adjacent voxels in each dimension
    new_size = [256, 256, 128]
    sitk_image = sitk.GetImageFromArray(data)
    for i in range(len(img_info[1])):
        if img_info[1][i] == 0:
            img_info[1][i] = 0.00001
    sitk_image.SetDirection(img_info[1])
    sitk_image.SetOrigin(img_info[2])
    sitk_image.SetSpacing(img_info[3])

    new_spacing = [(ospc * osz / nsz) for osz, ospc, nsz in
                   zip(sitk_image.GetSize(), sitk_image.GetSpacing(), new_size)]

    sitk_image = sitk.Resample(sitk_image, new_size, sitk.Transform(),
                               sitk.sitkLinear,
                               sitk_image.GetOrigin(),
                               new_spacing, sitk_image.GetDirection(),
                               0, sitk_image.GetPixelID())
    array = sitk.GetArrayFromImage(sitk_image)

    return array


def Resize(data, img_info):

    new_size = [256, 256, ]
    sitk_image = sitk.GetImageFromArray(data)
    for i in range(len(img_info[1])):
        if img_info[1][i] == 0:
            img_info[1][i] = 0.00001
    sitk_image.SetDirection(img_info[1])
    sitk_image.SetOrigin(img_info[2])
    sitk_image.SetSpacing(img_info[3])

    new_spacing = [(ospc * osz / nsz) for osz, ospc, nsz in
                   zip(sitk_image.GetSize(), sitk_image.GetSpacing(), new_size)]

    sitk_image = sitk.Resample(sitk_image, new_size, sitk.Transform(),
                               sitk.sitkLinear,
                               sitk_image.GetOrigin(),
                               new_spacing, sitk_image.GetDirection(),
                               0, sitk_image.GetPixelID())
    array = sitk.GetArrayFromImage(sitk_image)

    return array


class Lung_segmentation():  # 모듈 name을 class name으로 설정. Preprocessing 상속 **지우지 마세요
    def __init__(self):  # Parameter 탭에서 추가한 parameter 명을 key값으로 받도록 __init__에서 정의
        super(Lung_segmentation, self).__init__()  # 입력한 class name과 동일하게 super()에 입력

    def __call__(self, data, img_info=None, save_path=None):  # 이전 모듈의 hdf5 파일 하나씩 data라는 이름으로 call method의 input으로 들어옴
        img_array = data
        # get spacing
        raw_spacing = np.array(img_info[3])[[2, 1, 0]]
        # logging.info('spacing')
        # logging.info(raw_spacing)
        Sliceim, Mask = self.image_processing_segmentation_from_array_NEW(img_array, raw_spacing)
        data = Sliceim

        return data

    def image_processing_segmentation_from_array_NEW(self, imgarray, spacing):
        # dataloaded = np.load(wholepath)
        # imgarray = dataloaded['img']
        # spacing = dataloaded['spacing'][[2,1,0]]
        # labelarray = dataloaded['label']
        image_bi = self.binarize_per_slice(imgarray, spacing, intensity_th=-300, sigma=1, area_th=15,
                                           eccen_th=0.99)  # changes in new method
        flag = 0
        cut_num = 0
        cut_step = 2
        # margin = 5
        image0_bi = np.copy(image_bi)
        while flag == 0 and cut_num < image_bi.shape[0]:
            image_bi = np.copy(image0_bi)
            image_bi, flag = self.all_slice_analysis(image_bi, spacing, cut_num=cut_num, vol_limit=[0.68, 7.5],
                                                     area_th=6e3, dist_th=70)
            cut_num = cut_num + cut_step
        image_dsb = self.fill_hole(image_bi)
        # logging.info('image_dsb')
        # logging.info(image_dsb.shape)
        # logging.info('spacing')
        # logging.info(spacing)
        image1_dsb, image2_dsb, _ = self.two_lung_only(image_dsb, spacing)
        dm1 = self.process_mask(image1_dsb)
        dm2 = self.process_mask(image2_dsb)
        dilatedMask = dm1 + dm2
        Mask1 = image1_dsb + image2_dsb
        extramask = dilatedMask ^ Mask1
        # lung_mask1 = dilatedMask ^ extramask
        # lung_mask = np.asarray(lung_mask1[:, :, :], dtype=int)
        ww_array = self.windowW_windowL(imgarray, 1000, -650)
        bone_thresh = 210
        pad_value = 0
        sliceim = ww_array * dilatedMask + pad_value * (1 - dilatedMask).astype('uint8')
        # bones = sliceim * extramask > bone_thresh
        # sliceim[bones] = 500
        return sliceim, dilatedMask

    def fill_hole(self, bw):
        # fill 3d holes
        label = measure.label(~bw)
        # idendify corner components
        bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1],label[-1, 0, 0], label[-1, 0, -1], label[-1, -1, 0], label[-1, -1, -1]])
        bw = ~np.in1d(label, list(bg_label)).reshape(label.shape)

        return bw

    def all_slice_analysis(self, bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=6e3, dist_th=62):
        # in some cases, several top layers need to be removed first
        if cut_num > 0:
            bw0 = np.copy(bw)
            bw[-cut_num:] = False
        label = measure.label(bw, connectivity=1)
        # remove components access to corners
        mid = int(label.shape[2] / 2)
        bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], label[-1 - cut_num, 0, 0], label[-1 - cut_num, 0, -1], label[-1 - cut_num, -1, 0],
                        label[-1 - cut_num, -1, -1], label[0, 0, mid], label[0, -1, mid], label[-1 - cut_num, 0, mid], label[-1 - cut_num, -1, mid]])
        for l in bg_label:
            label[label == l] = 0

        # select components based on volume
        properties = measure.regionprops(label)
        for prop in properties:
            if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
                label[label == prop.label] = 0

        # prepare a distance map for further analysis
        x_axis = np.linspace(-label.shape[1] / 2 + 0.5, label.shape[1] / 2 - 0.5, label.shape[1]) * spacing[1]
        y_axis = np.linspace(-label.shape[2] / 2 + 0.5, label.shape[2] / 2 - 0.5, label.shape[2]) * spacing[2]
        x, y = np.meshgrid(x_axis, y_axis)
        d = (x ** 2 + y ** 2) ** 0.5
        vols = measure.regionprops(label)
        valid_label = set()
        # select components based on their area and distance to center axis on all slices
        for vol in vols:
            single_vol = label == vol.label
            slice_area = np.zeros(label.shape[0])
            min_distance = np.zeros(label.shape[0])
            for i in range(label.shape[0]):
                slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
                min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))

            if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
                valid_label.add(vol.label)
        if valid_label == 0:
            print('lung segmentation failed,please check data')
        bw = np.in1d(label, list(valid_label)).reshape(label.shape)

        # fill back the parts removed earlier
        if cut_num > 0:
            # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
            bw1 = np.copy(bw)
            bw1[-cut_num:] = bw0[-cut_num:]
            bw2 = np.copy(bw)
            bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
            bw3 = bw1 & bw2
            label = measure.label(bw, connectivity=1)
            label3 = measure.label(bw3, connectivity=1)
            l_list = list(set(np.unique(label)) - {0})
            valid_l3 = set()
            for l in l_list:
                indices = np.nonzero(label == l)
                l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
                if l3 > 0:
                    valid_l3.add(l3)
            bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)

        return bw, len(valid_label)

    def binarize_per_slice(self, image, spacing, intensity_th=-600, sigma=1, area_th=30, eccen_th=0.99,
                           bg_patch_size=10):
        bw = np.zeros(image.shape, dtype=bool)

        # prepare a mask, with all corner values set to nan
        image_size = image.shape[1]
        grid_axis = np.linspace(-image_size / 2 + 0.5, image_size / 2 - 0.5, image_size)
        x, y = np.meshgrid(grid_axis, grid_axis)
        d = (x ** 2 + y ** 2) ** 0.5
        nan_mask = (d < image_size / 2).astype(float)
        nan_mask[nan_mask == 0] = np.nan
        for i in range(image.shape[0]):
            # Check if corner pixels are identical, if so the slice  before Gaussian filtering
            if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
                current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask),
                                                                   sigma,
                                                                   truncate=2.0) < intensity_th
            else:
                current_bw = scipy.ndimage.filters.gaussian_filter(image[i].astype('float32'), sigma,
                                                                   truncate=2.0) < intensity_th

            # select proper components
            label = measure.label(current_bw)
            properties = measure.regionprops(label)
            valid_label = set()
            for prop in properties:
                if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
                    valid_label.add(prop.label)
            current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
            bw[i] = current_bw
        return bw

    def two_lung_only(self, bw, spacing, max_iter=22, max_ratio=4.8):
        def extract_main(bw, cover=0.95):
            for i in range(bw.shape[0]):
                current_slice = bw[i]
                label = measure.label(current_slice)
                properties = measure.regionprops(label)
                properties.sort(key=lambda x: x.area, reverse=True)
                area = [prop.area for prop in properties]
                count = 0
                sum = 0
                while sum < np.sum(area) * cover:
                    sum = sum + area[count]
                    count = count + 1
                filter = np.zeros(current_slice.shape, dtype=bool)
                for j in range(count):
                    bb = properties[j].bbox
                    filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2], bb[1]:bb[3]] | properties[j].convex_image
                bw[i] = bw[i] & filter

            label = measure.label(bw)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            bw = label == properties[0].label
            return bw

        def fill_2d_hole(bw):
            for i in range(bw.shape[0]):
                current_slice = bw[i]
                label = measure.label(current_slice)
                properties = measure.regionprops(label)
                for prop in properties:
                    bb = prop.bbox
                    current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[bb[0]:bb[2],
                                                              bb[1]:bb[3]] | prop.filled_image
                bw[i] = current_slice

            return bw

        found_flag = False
        iter_count = 0
        bw0 = np.copy(bw)
        while not found_flag and iter_count < max_iter:
            label = measure.label(bw, connectivity=2)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            if len(properties) > 1 and properties[0].area / properties[1].area < max_ratio:
                found_flag = True
                bw1 = label == properties[0].label
                bw2 = label == properties[1].label
            else:
                bw = scipy.ndimage.binary_erosion(bw)
                iter_count = iter_count + 1

        if found_flag:
            d1 = scipy.ndimage.morphology.distance_transform_edt(bw1 == False, sampling=spacing)
            d2 = scipy.ndimage.morphology.distance_transform_edt(bw2 == False, sampling=spacing)
            bw1 = bw0 & (d1 < d2)
            bw2 = bw0 & (d1 > d2)

            bw1 = extract_main(bw1)
            bw2 = extract_main(bw2)

        else:
            bw1 = bw0
            bw2 = np.zeros(bw.shape).astype('bool')

        bw1 = fill_2d_hole(bw1)
        bw2 = fill_2d_hole(bw2)
        bw = bw1 | bw2

        return bw1, bw2, bw

    def process_mask(self, mask):
        convex_mask = np.copy(mask)
        for i_layer in range(convex_mask.shape[0]):
            mask1 = np.ascontiguousarray(mask[i_layer])
            if np.sum(mask1) > 0:
                mask2 = convex_hull_image(mask1)
                if np.sum(mask2) > 1.5 * np.sum(mask1):
                    mask2 = mask1
            else:
                mask2 = mask1
            convex_mask[i_layer] = mask2
        struct = generate_binary_structure(3, 1)
        dilatedMask = binary_dilation(convex_mask, structure=struct, iterations=10)
        return dilatedMask

    def windowW_windowL(self, img, ww, wl):
        img[np.where(img == 170)] = -2000
        upper = wl + ww / 2
        lower = wl - ww / 2
        img = (img - lower) / (upper - lower)
        img[img < 0] = 0
        img[img > 1] = 1
        return (img * 255).astype(np.uint8)

class FeatureMerge():
    def __init__(self):
        super(FeatureMerge, self).__init__()

    def __call__(self, *data, save_path=None):
        data_list = data # input data list
        new_data = data_list[0]
        overlay = data_list[1]
        new_array = np.copy(new_data)

        overlay_array = np.copy(overlay)
        overlay_array = (2048 * (overlay_array/255) - 1024).astype(np.int16)

        n3 = np.min(overlay_array)
        # main merge algorithm
        new_data = np.where(overlay_array != n3, new_array, 0)

        return new_data










