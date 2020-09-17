

g_slice_thickness = 3.0

class Parameter(object):
    def __init__(
            self,
            epochs,
            sos_epochs,
            lr,
            dropout,
            depth,
            height,
            width,
            sol_format,
            sos_format,
            ww,
            wl,
            se_size,
            bbox):
        self.epochs = epochs
        self.sos_epochs = sos_epochs
        self.lr = lr
        self.dropout = dropout
        self.depth = depth
        self.height = height
        self.width = width
        self.sol_format = sol_format
        self.sos_format = sos_format
        self.ww = ww
        self.wl = wl
        self.se_size = se_size
        self.bbox = bbox
        pass

# dict_supper_parameters = {
#     # 'Eye_L': Parameter(epochs=8, lr=0.001, dropout=0.8, depth=160, height=96, width=96, sol_format='dicom8bit',
#     #                    sos_format='dicom8bit', ww=350, wl=40, se_size=2, bbox=[16, 128, 128]),
#     # 'Eye_R': Parameter(epochs=8, lr=0.001, dropout=0.8, depth=160, height=96, width=96, sol_format='dicom8bit',
#     #                    sos_format='dicom8bit', ww=350, wl=40, se_size=2, bbox=[16, 128, 128]),
#     # 'Len_L': Parameter(epochs=8, sos_epochs=13, lr=0.001, dropout=0.8, depth=160, height=96, width=96, sol_format='dicom8bit',
#     #                    sos_format='dicom8bit', ww=100, wl=35, se_size=6, bbox=[16, 64, 64]),
#     # 'Len_R': Parameter(epochs=8, lr=0.001, dropout=0.8, depth=160, height=96, width=96, sol_format='dicom8bit',
#     #                    sos_format='dicom', ww=350, wl=60, se_size=6, bbox=[16, 64, 64]),
#     # 'Optic_Nerve_L': Parameter(epochs=8, lr=0.001, dropout=0.8, depth=160, height=96, width=96, sol_format='dicom8bit',
#     #                            sos_format='dicom', ww=400, wl=100, se_size=4, bbox=[16, 128, 128]),
#     # 'Optic_Nerve_R': Parameter(epochs=8, lr=0.001, dropout=0.8, depth=160, height=96, width=96, sol_format='dicom8bit',
#     #                            sos_format='dicom', ww=400, wl=100, se_size=4, bbox=[16, 128, 128]),
#     # 'Optic_Chiasm': Parameter(epochs=8, lr=0.001, dropout=0.8, depth=160, height=96, width=96, sol_format='dicom8bit',
#     #                           sos_format='dicom8bit', ww=530, wl=240, se_size=4, bbox=[16, 128, 128]),
#     # 'Brainstem': Parameter(epochs=8, lr=0.001, dropout=0.8, depth=160, height=96, width=96, sol_format='dicom8bit',
#     #                        sos_format='dicom8bit', ww=200, wl=100, se_size=2, bbox=[32, 256, 256]),
#     # 'Parotid_Gland_L': Parameter(epochs=8, lr=0.001, dropout=0.8, depth=160, height=96, width=96,
#     #                              sol_format='dicom8bit', sos_format='dicom', ww=130, wl=35, se_size=2,
#     #                              bbox=[32, 256, 256]),
#     # 'Parotid_Gland_R': Parameter(epochs=8, lr=0.001, dropout=0.8, depth=160, height=96, width=96,
#     #                              sol_format='dicom8bit', sos_format='dicom', ww=130, wl=35, se_size=2,
#     #                              bbox=[32, 256, 256]),
#     # 'Inner_Ear_L': Parameter(epochs=8, lr=0.001, dropout=0.8, depth=160, height=96, width=96, sol_format='dicom8bit',
#     #                          sos_format='dicom8bit', ww=4000, wl=700, se_size=2, bbox=[16, 128, 128]),
#     # 'Inner_Ear_R': Parameter(epochs=8, lr=0.001, dropout=0.8, depth=160, height=96, width=96, sol_format='dicom8bit',
#     #                          sos_format='dicom8bit', ww=4000, wl=700, se_size=2, bbox=[16, 128, 128]),
#     # 'Middle_Ear_L': Parameter(epochs=8, lr=0.001, dropout=0.8, depth=160, height=96, width=96, sol_format='dicom8bit',
#     #                           sos_format='dicom8bit', ww=4000, wl=700, se_size=2, bbox=[32, 256, 256]),
#     # 'Middle_Ear_R': Parameter(epochs=8, lr=0.001, dropout=0.8, depth=160, height=96, width=96, sol_format='dicom8bit',
#     #                           sos_format='dicom8bit', ww=4000, wl=700, se_size=2, bbox=[32, 256, 256]),
#     # 'Mandible_Joint_L': Parameter(epochs=8, lr=0.001, dropout=0.8, depth=160, height=96, width=96,
#     #                               sol_format='dicom8bit', sos_format='dicom8bit', ww=1600, wl=450, se_size=2,
#     #                               bbox=[16, 128, 128]),
#     # 'Mandible_Joint_R': Parameter(epochs=8, lr=0.001, dropout=0.8, depth=160, height=96, width=96,
#     #                               sol_format='dicom8bit', sos_format='dicom8bit', ww=1600, wl=450, se_size=2,
#     #                               bbox=[16, 128, 128]),
#     # 'Pituitary': Parameter(epochs=8, lr=0.001, dropout=0.8, depth=160, height=96, width=96, sol_format='dicom8bit',
#     #                        sos_format='dicom', ww=320, wl=120, se_size=6, bbox=[16, 64, 64]),
#     # 'Temporal_Lobe_L': Parameter(epochs=8, lr=0.001, dropout=0.8, depth=160, height=96, width=96,
#     #                              sol_format='dicom8bit', sos_format='dicom', ww=320, wl=120, se_size=2,
#     #                              bbox=[32, 256, 256]),
#     # 'Temporal_Lobe_R': Parameter(epochs=8, lr=0.001, dropout=0.8, depth=160, height=96, width=96,
#     #                              sol_format='dicom8bit', sos_format='dicom', ww=320, wl=120, se_size=2,
#     #                              bbox=[32, 256, 256]),
#     # 'Mandible_L': Parameter(epochs=8, lr=0.001, dropout=0.8, depth=160, height=96, width=96, sol_format='dicom8bit',
#     #                         sos_format='dicom', ww=250, wl=40, se_size=2, bbox=[48, 256, 256]),
#     # 'Mandible_R': Parameter(epochs=8, lr=0.001, dropout=0.8, depth=160, height=96, width=96, sol_format='dicom8bit',
#     #                         sos_format='dicom', ww=250, wl=40, se_size=2, bbox=[48, 256, 256]),
# #     'Spinal_Cord' : Parameter(epochs=8, lr=0.001, dropout=0.8, depth=160, height=96, width=96, sol_format='dicom8bit', sos_format='dicom', ww=250, wl=40, se_size=3, bbox=[96, 128, 128]),
# }

dict_supper_parameters = {
    # 'Len_L': Parameter(epochs=6, sos_epochs=12, lr=0.001, dropout=0.7, depth=64, height=128, width=128,
    #                    sol_format='dicom8bit', sos_format='dicom8bit', ww=160, wl=40, se_size=6, bbox=[16, 128, 128]),
    # 'Len_R': Parameter(epochs=6, sos_epochs=12, lr=0.001, dropout=0.7, depth=64, height=128, width=128,
    #                    sol_format='dicom8bit', sos_format='dicom8bit', ww=160, wl=40, se_size=6, bbox=[16, 128, 128]),

    # 'Eye_L': Parameter(epochs=4, sos_epochs=8, lr=0.001, dropout=0.7, depth=80, height=128, width=128,
    #                    sol_format='dicom8bit', sos_format='dicom8bit', ww=160, wl=40, se_size=2, bbox=[32, 128, 128]),
    # 'Eye_R': Parameter(epochs=4, sos_epochs=8, lr=0.001, dropout=0.7, depth=80, height=128, width=128,
    #                    sol_format='dicom8bit', sos_format='dicom8bit', ww=160, wl=40, se_size=2, bbox=[32, 128, 128]),
    # 'Optic_Nerve_L': Parameter(epochs=4, sos_epochs=11, lr=0.001, dropout=0.7, depth=64, height=128, width=128,
    #                            sol_format='dicom8bit',sos_format='dicom8bit',ww=350, wl=40, se_size=4,bbox=[16, 128, 128]),

    # 'Optic_Nerve_R': Parameter(epochs=6, sos_epochs=11, lr=0.001, dropout=0.7, depth=64, height=128, width=128, sol_format='dicom8bit', sos_format='dicom8bit', ww=350, wl=40, se_size=4, bbox=[16, 128, 128]),
    #'Optic_Chiasm': Parameter(epochs=3, sos_epochs=12, lr=0.001, dropout=0.8, depth=64, height=128, width=128, sol_format='dicom8bit', sos_format='dicom8bit', ww=150, wl=90, se_size=4, bbox=[16, 128, 128]),
    # 'Brainstem': Parameter(epochs=3, sos_epochs=5, lr=0.001, dropout=0.7, depth=64, height=128, width=128, sol_format='dicom8bit', sos_format='dicom8bit', ww=200, wl=100, se_size=0, bbox=[48, 96, 96]),
    'Parotid_Gland_L': Parameter(epochs=3, sos_epochs=8, lr=0.001, dropout=0.7, depth=64, height=128, width=128, sol_format='dicom8bit', sos_format='dicom8bit', ww=450, wl=10, se_size=0, bbox=[48, 128, 128]),
    'Parotid_Gland_R': Parameter(epochs=3, sos_epochs=8, lr=0.001, dropout=0.7, depth=64, height=128, width=128, sol_format='dicom8bit', sos_format='dicom8bit', ww=450, wl=10, se_size=0, bbox=[48, 128, 128]),
    #'Inner_Ear_L': Parameter(epochs=3, sos_epochs=12, lr=0.001, dropout=0.8, depth=64, height=128, width=128, sol_format='dicom8bit', sos_format='dicom8bit', ww=4000, wl=700, se_size=4, bbox=[16, 128, 128]),
    #'Inner_Ear_R': Parameter(epochs=3, sos_epochs=12, lr=0.001, dropout=0.8, depth=64, height=128, width=128, sol_format='dicom8bit', sos_format='dicom8bit', ww=4000, wl=700, se_size=4, bbox=[16, 128, 128]),
    #'Middle_Ear_L': Parameter(epochs=3, sos_epochs=12, lr=0.001, dropout=0.8, depth=64, height=128, width=128, sol_format='dicom8bit', sos_format='dicom8bit', ww=1600, wl=450, se_size=3, bbox=[32, 128, 128]),
    #'Middle_Ear_R': Parameter(epochs=3, sos_epochs=12, lr=0.001, dropout=0.8, depth=64, height=128, width=128, sol_format='dicom8bit', sos_format='dicom8bit', ww=1600, wl=450, se_size=2, bbox=[32, 128, 128]),
    #'Mandible_Joint_L': Parameter(epochs=3, sos_epochs=12, lr=0.001, dropout=0.8, depth=64, height=128, width=128, sol_format='dicom8bit', sos_format='dicom8bit', ww=1600, wl=450, se_size=2, bbox=[16, 128, 128]),
    #'Mandible_Joint_R': Parameter(epochs=3, sos_epochs=12, lr=0.001, dropout=0.8, depth=64, height=128, width=128, sol_format='dicom8bit', sos_format='dicom8bit', ww=1600, wl=450, se_size=2, bbox=[16, 128, 128]),
    # 'Pituitary': Parameter(epochs=5, sos_epochs=6, lr=0.001, dropout=0.7, depth=64, height=128, width=128,
    #                        sol_format='dicom8bit', sos_format='dicom8bit', ww=320, wl=120, se_size=6, bbox=[16, 128, 128]),
    #'Temporal_Lobe_L': Parameter(epochs=3, sos_epochs=6, lr=0.001, dropout=0.8, depth=64, height=128, width=128, sol_format='dicom8bit', sos_format='dicom8bit', ww=180, wl=110, se_size=0, bbox=[32, 256, 256]),
    #'Temporal_Lobe_R': Parameter(epochs=3, sos_epochs=12, lr=0.001, dropout=0.8, depth=64, height=128, width=128,
    #                             sol_format='dicom8bit', sos_format='dicom8bit', ww=180, wl=110, se_size=0, bbox=[32, 256, 256]),
    # 'Mandible_L': Parameter(epochs=3, sos_epochs=8, lr=0.001, dropout=0.7, depth=64, height=128, width=128, sol_format='dicom8bit', sos_format='dicom8bit', ww=1600, wl=450, se_size=0, bbox=[64, 128, 128]),
    # 'Mandible_R': Parameter(epochs=3, sos_epochs=8, lr=0.001, dropout=0.7, depth=64, height=128, width=128, sol_format='dicom8bit', sos_format='dicom8bit', ww=1600, wl=450, se_size=0, bbox=[64, 128, 128]),
    # 'Spinal_Cord': Parameter(epochs=4, sos_epochs=11, lr=0.001, dropout=0.7, depth=64, height=128, width=128, sol_format='dicom8bit', sos_format='dicom8bit', ww=250, wl=40, se_size=2, bbox=[96, 128, 128]),
}

kernel_size_x = kernel_size_y = kernel_size_z = 3
kernel_shape_parameters1 = [
    [kernel_size_z, kernel_size_y, kernel_size_x, 1, 8],  # 0 layer0
    [kernel_size_z, kernel_size_y, kernel_size_x, 8, 8],  # 1 layer1
    [kernel_size_z, kernel_size_y, kernel_size_x, 8, 16],  # 2 down1
    [kernel_size_z, kernel_size_y, kernel_size_x, 16, 16],  # 3 layer2_1
    [kernel_size_z, kernel_size_y, kernel_size_x, 16, 16],  # 4 layer2_2
    [kernel_size_z, kernel_size_y, kernel_size_x, 16, 32],  # 5 down2
    [kernel_size_z, kernel_size_y, kernel_size_x, 32, 32],  # 6 layer3_1
    [kernel_size_z, kernel_size_y, kernel_size_x, 32, 32],  # 7 layer3_2
    [kernel_size_z, kernel_size_y, kernel_size_x, 32, 32],  # 8 layer3_3
    [kernel_size_z, kernel_size_y, kernel_size_x, 32, 64],  # 9 down3
    [kernel_size_z, kernel_size_y, kernel_size_x, 64, 64],  # 10 layer4_1
    [kernel_size_z, kernel_size_y, kernel_size_x, 64, 64],  # 11 layer4_2
    [kernel_size_z, kernel_size_y, kernel_size_x, 64, 64],  # 12 layer4_3
    [kernel_size_z, kernel_size_y, kernel_size_x, 64, 128],  # 13 down4
    [kernel_size_z, kernel_size_y, kernel_size_x, 128, 128],  # 14 layer5_1
    [kernel_size_z, kernel_size_y, kernel_size_x, 128, 128],  # 15 layer5_2
    [kernel_size_z, kernel_size_y, kernel_size_x, 128, 128],  # 16 layer5_3
    [1, 1, 1, 128, 64],  # 17 g1
    [kernel_size_z, kernel_size_y, kernel_size_x, 64, 128],  # 18 deconv1
    [kernel_size_z, kernel_size_y, kernel_size_x, 128, 64],  # 19 layer6_1
    [kernel_size_z, kernel_size_y, kernel_size_x, 64, 64],  # 20 layer6_2
    [kernel_size_z, kernel_size_y, kernel_size_x, 64, 64],  # 21 layer6_3
    [1, 1, 1, 64, 32],  # 22 g2
    [kernel_size_z, kernel_size_y, kernel_size_x, 32, 64],  # 23 deconv2
    [kernel_size_z, kernel_size_y, kernel_size_x, 64, 32],  # 24 layer7_1
    [kernel_size_z, kernel_size_y, kernel_size_x, 32, 32],  # 25 layer7_2
    [1, 1, 1, 32, 16],  # 26 g3
    [kernel_size_z, kernel_size_y, kernel_size_x, 16, 32],  # 27 deconv3
    [kernel_size_z, kernel_size_y, kernel_size_x, 32, 16],  # 28 layer8_1
    [kernel_size_z, kernel_size_y, kernel_size_x, 16, 16],  # 29 layer8_2
    [kernel_size_z, kernel_size_y, kernel_size_x, 16, 16],  # 30 layer8_3
    [1, 1, 1, 16, 8],  # 31 g4
    [kernel_size_z, kernel_size_y, kernel_size_x, 8, 16],  # 32 deconv4
    [kernel_size_z, kernel_size_y, kernel_size_x, 16, 16],  # 33 layer9_1
    [kernel_size_z, kernel_size_y, kernel_size_x, 16, 16],  # 34 layer9_2
    [kernel_size_z, kernel_size_y, kernel_size_x, 16, 16],  # 35 layer9_3
    [kernel_size_z, kernel_size_y, kernel_size_x, 16, 1]  # 36 output
]

kernel_shape_parameters2 = [
    [kernel_size_z, kernel_size_y, kernel_size_x, 1, 16],  # 0 layer0
    [kernel_size_z, kernel_size_y, kernel_size_x, 16, 16],  # 1 layer1
    [kernel_size_z, kernel_size_y, kernel_size_x, 16, 32],  # 2 down1
    [kernel_size_z, kernel_size_y, kernel_size_x, 32, 32],  # 3 layer2_1
    [kernel_size_z, kernel_size_y, kernel_size_x, 32, 32],  # 4 layer2_2
    [kernel_size_z, kernel_size_y, kernel_size_x, 32, 64],  # 5 down2
    [kernel_size_z, kernel_size_y, kernel_size_x, 64, 64],  # 6 layer3_1
    [kernel_size_z, kernel_size_y, kernel_size_x, 64, 64],  # 7 layer3_2
    [kernel_size_z, kernel_size_y, kernel_size_x, 64, 64],  # 8 layer3_3
    [kernel_size_z, kernel_size_y, kernel_size_x, 64, 128],  # 9 down3
    [kernel_size_z, kernel_size_y, kernel_size_x, 128, 128],  # 10 layer4_1
    [kernel_size_z, kernel_size_y, kernel_size_x, 128, 128],  # 11 layer4_2
    [kernel_size_z, kernel_size_y, kernel_size_x, 128, 128],  # 12 layer4_3
    [kernel_size_z, kernel_size_y, kernel_size_x, 128, 256],  # 13 down4
    [kernel_size_z, kernel_size_y, kernel_size_x, 256, 256],  # 14 layer5_1
    [kernel_size_z, kernel_size_y, kernel_size_x, 256, 256],  # 15 layer5_2
    [kernel_size_z, kernel_size_y, kernel_size_x, 256, 256],  # 16 layer5_3
    [1, 1, 1, 256, 128],  # 17 g1
    [kernel_size_z, kernel_size_y, kernel_size_x, 128, 256],  # 18 deconv1
    [kernel_size_z, kernel_size_y, kernel_size_x, 256, 128],  # 19 layer6_1
    [kernel_size_z, kernel_size_y, kernel_size_x, 128, 128],  # 20 layer6_2
    [kernel_size_z, kernel_size_y, kernel_size_x, 128, 128],  # 21 layer6_3
    [1, 1, 1, 128, 64],  # 22 g2
    [kernel_size_z, kernel_size_y, kernel_size_x, 64, 128],  # 23 deconv2
    [kernel_size_z, kernel_size_y, kernel_size_x, 128, 64],  # 24 layer7_1
    [kernel_size_z, kernel_size_y, kernel_size_x, 64, 64],  # 25 layer7_2
    [1, 1, 1, 64, 32],  # 26 g3
    [kernel_size_z, kernel_size_y, kernel_size_x, 32, 64],  # 27 deconv3
    [kernel_size_z, kernel_size_y, kernel_size_x, 64, 32],  # 28 layer8_1
    [kernel_size_z, kernel_size_y, kernel_size_x, 32, 32],  # 29 layer8_2
    [kernel_size_z, kernel_size_y, kernel_size_x, 32, 32],  # 30 layer8_3
    [1, 1, 1, 32, 16],  # 31 g4
    [kernel_size_z, kernel_size_y, kernel_size_x, 16, 32],  # 32 deconv4
    [kernel_size_z, kernel_size_y, kernel_size_x, 32, 32],  # 33 layer9_1
    [kernel_size_z, kernel_size_y, kernel_size_x, 32, 32],  # 34 layer9_2
    [kernel_size_z, kernel_size_y, kernel_size_x, 32, 32],  # 35 layer9_3
    [kernel_size_z, kernel_size_y, kernel_size_x, 32, 1]  # 36 output
]