import struct
import zlib

class PNG:
    def __init__(self,inputfile):
        self.f = open(inputfile,'rb')
        self.imgdata = self.f.read()
        #書き込み用にsignatureとIHDRを読み出しておく
        self.head = struct.unpack_from(">33s", self.imgdata, 0)
        #PNG画像か否か判断。PNG画像であれば各種データ読み出し。
        if struct.unpack_from(">3s", self.imgdata, 1) == (b'PNG',):
            self.i_width = struct.unpack_from(">I", self.imgdata, 16)
            self.i_height = struct.unpack_from(">I", self.imgdata, 20)
            self.bit_depth = struct.unpack_from(">B", self.imgdata, 24)
            self.color_type = struct.unpack_from(">B", self.imgdata, 25)
            self.comp_method = struct.unpack_from(">B", self.imgdata, 26)
            self.filter_method = struct.unpack_from(">B", self.imgdata, 27)
            self.interlace_method = struct.unpack_from(">B", self.imgdata, 28)
            self.crc = struct.unpack_from(">B", self.imgdata, 29)
            #IDATの読み出し。複数ある場合全て連結する。はIENDが現れるまで繰り返す
            self.count = 30
            self.idata_type = struct.unpack_from(">4s", self.imgdata, self.count)
            self.img_length = 0 #IDATの合計データ長
            self.img_data = b'' #IDATのデータ部が入る
            self.cnt = 0        #IDATチャンクの数を数える
            while self.idata_type != (b'IEND',):
                self.idata_type = struct.unpack_from(">4s", self.imgdata, self.count)
                if self.idata_type == (b'IDAT',):
                    self.idata_length = struct.unpack_from(">I",self.imgdata,self.count-4)
                    self.img_length += self.idata_length[0]
                    self.img_subdata = struct.unpack_from(">"+str(self.idata_length[0])+"s",self.imgdata,self.count+4)
                    self.img_data += self.img_subdata[0]
                    self.cnt += 1
                self.count += 1
            print('read OK','This Image is',self.count,'byte')
        else:
            print('This file is not PNG image')
        self.f.close()
    def outputPNG(self,outputfile):
        self.ff=open(outputfile,'wb')
        self.ff.write(struct.pack(">33s",self.head[0]))
        self.ff.write(struct.pack(">I",self.img_length))
        self.ff.write(struct.pack(">4s",b'IDAT'))
        self.ff.write(struct.pack(">"+str(self.img_length)+"s",self.img_data))
        self.ff.write(struct.pack(">I",zlib.crc32(b'IDAT' + self.img_data)))
        self.ff.write(struct.pack(">I",0))
        self.ff.write(struct.pack(">4s",b'IEND'))
        self.ff.write(struct.pack(">I",zlib.crc32(b'IEND')))
        self.ff.close()
    def printDATA(self):
        print('bit_depth =',self.bit_depth[0],'color_type =',self.color_type[0],
            'comp_method =',self.comp_method[0],'filter_method =',self.filter_method[0],
            'interlace_method =',self.interlace_method[0],'crc =',self.crc[0])
        print('width =',self.i_width[0],', height =',self.i_height[0])
        print('image data length =',self.img_length,' byte','IDAT THUNK cnt =',self.cnt)
    def getIDAT(self):
        return self.img_data
    def searchTNK(self,thunk):
        self.dmy1 = 0
        self.dmy2 = 0
        self.thunk_type = (b'',)
        while self.thunk_type != (b'IEND',):
            self.thunk_type = struct.unpack_from(">4s", self.imgdata, self.dmy1)
            if self.thunk_type == (thunk,):
                self.thunk_length = struct.unpack_from(">I",self.imgdata,self.dmy1-4)
                self.thunk_value = struct.unpack_from(">B",self.imgdata,self.dmy1+4)
                print(self.thunk_type[0],'length=',self.thunk_length[0],'value=',self.thunk_value[0])
                self.dmy2 += 1
            self.dmy1 += 1
        if self.dmy2 == 0:
            print('no thunk',thunk)


def unitTest(src_path, dst_path):
    from tqdm import tqdm
    label_set = os.listdir(src_path)
    dst_file = os.path.join(dst_path, 'out.png')
    for k in tqdm(iterable=label_set, desc='From '+src_path):
        src_file = os.path.join(src_path, k)
        png = PNG(src_file)
        png.printDATA()
        # png.outputPNG(dst_file)
        idat = png.getIDAT()
        # print("png.getIDAT(dst_file): {}".format(idat))
        #FIXME: get image data, expected to [0, 1, 2, ..., 22].
        png.searchTNK(b'sRGB')
        png.searchTNK(b'PLTE')
        break
    pass

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    import os
    from configs import cfg_factory
    cfg = cfg_factory['bisenetv2']
    unitTest(cfg.train_img_anns, cfg.respth)
    pass