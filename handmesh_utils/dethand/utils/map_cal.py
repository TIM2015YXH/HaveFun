"""
Mean Average Precision
~~~~~~~~~~~~~
by HYK
~~~~~~~~~~~~~
This file is a tool file for calculating Mean Average Precision. requires numpy
"""

import numpy as np

#hyper parameters

class CalcMAP():
    def __init__(self, classes, width=256, height=256, overlap_thresh=0.5 ):
        """

        :param classes: hand classes, list
        :param width: image width, int
        :param height: image height, int
        :param overlap_thresh: overlap threshold, [0, 1], float
        """

        self.classes = classes
        self.overlap_thresh = overlap_thresh
        self.width = width
        self.height = height
        #self.padding = padding
        self.class_to_idx = dict(zip(classes, range(0, len(classes))))
        self.prec_rec_ap_result = {}
        self.imgname2GT = {}
        self.detect_results = {}

    def set_overlap_thresh(self, overlap_thresh):
        """
        set the over threshold
        :param overlap_thresh: overlap threshold, float
        :return: None
        """
        self.overlap_thresh = overlap_thresh

    def do_python_eval(self,  use_07=False):
        """
        Caluculate the MAP on the val datasets
        Arguments:
            output_dir: (type: str), Save the last map
            result_cache: (type: str), Ouput the prediction result
            valpath: (type: str), val dataset path
            use_07:(type: bool), whether use 07 voc standard

        """
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = use_07
        #print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))



        for class_index, class_name in enumerate(self.classes):
            rec, prec, ap = self.eval_per_class(class_name,
                                 self.overlap_thresh, use_07_metric=use_07)

            aps += [ap]
            #print('AP for {} = {:.4f}'.format(class_name, ap))
            #print('rec for {} = {:.4f}'.format(class_name, rec[-1]))
            #print('prec for {} = {:.4f}'.format(class_name, prec[-1]))
            self.prec_rec_ap_result[class_name] = {'rec': rec, 'prec': prec, 'ap': ap}

        #print('Mean AP = {:.4f}'.format(np.mean(aps)))
        #print('~~~~~~~~')
        #print('Results:')
        #for ap in aps:
            #print('{:.3f}'.format(ap))
        #print("Final result \n")
        #print('{:.3f}'.format(np.mean(aps)))
        #print('~~~~~~~~')

        return np.mean(aps),rec[-1],prec[-1]

    def class_ap(self, rec, prec, use_07_metric=False):
        """
        ap = class_ap(rec, prec, [use_07_metric])
        Compute AP given precision and recall.
        If use_07_metric is true, uses the VOC 07 11 point method (default:True).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap



    def eval_per_class(self,  classname,
                 overlap_thresh=0.5, use_07_metric=False):
        """
        rec, prec, ap = eval_per_class(detpath, annopath, imagesetfile, classname,
                                     [overlap_thresh],[use_07_metric])
        Top level function that does the PASCAL VOC evaluation.

        Arguments:
            detectionPath: (type: str) Path to detections detectionPath.format(classname) should produce the detection results file.
            valpath: (type: str) Path to val dataset path(.txt file)
            classname: (type: str) Category name, e.g, 'sheep'
            result_cache: Directory for caching the annotations
            [overlap_thresh]: Overlap threshold (default = 0.5)
            [use_07_metric]: Whether to use VOC07's 11 point AP computation (default True)
    """
        # assumes detections are in detpath.format(classname)
        # assumes annotations are in annopath.format(imagename)
        # assumes imagesetfile is a text file with each line an image name
        # cachedir caches the annotations in a pickle file
        # first load gt

        # extract gt objects for this class
        imgnames = self.imgname2GT.keys()
        class2GT = {}
        npos = 0

        for imagename in imgnames:
            R = [obj for obj in self.imgname2GT[imagename] if obj[0] == classname]

            if len(R) < 1:
                continue
            # Gain box
            bbox = np.array([x[1:] for x in R])
            # set `difficult` as 0
            difficult = np.array([0 for x in R]).astype(np.bool)
            # print ('difficult: ', difficult)
            det = [0] * len(R)
            det = np.array(det)
            npos = npos + sum(~difficult)
            # print ('npos: ', npos)
            class2GT[imagename] = {'bbox': bbox,
                                   'difficult': difficult,
                                   'det': det}


        splitlines = self.detect_results[classname]
        if any(splitlines) == 1:
            # x: [imgname, score, xmin, ymin, xmax, ymax]
            # splitlines = [x.strip().split(' ') for x in lines]
            imgnames = [x[0] for x in splitlines]

            confidence = np.array([float(x[1]) for x in splitlines])
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            BB = BB[sorted_ind, :]
            sortedimgnames = [imgnames[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            nd = len(sortedimgnames)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            ind = 0
            for d in range(nd):
                ovmax = 0
                # sortedimgnames[d] = sortedimgnames[d].replace('/media/disk1/wangzairan/dataset/hand/yukunhu/hand/', '/media/disk1/huyukun/data_standard/hand/')
                if not sortedimgnames[d] in class2GT:
                    continue
                R = class2GT[sortedimgnames[d]]
                # sortedimgnames[d] = sortedimgnames[d].replace('/media/disk1/huyukun/data_standard/hand/','/media/disk1/wangzairan/dataset/hand/yukunhu/hand/')
                # imgHeight, imgWidth, _ = cv2.imread(sortedimgnames[d]).shape

                imgHeight = self.height
                imgWidth = self.width

                bb = BB[d, :].astype(float)

                BBGT = R['bbox'].astype(float)
                BBdet = R['det']
                if BBGT.size > 0:
                    # compute overlaps intersection(IoU)
                    ixmin = np.maximum(BBGT[:, 0] * imgWidth, bb[0]* imgWidth)
                    iymin = np.maximum(BBGT[:, 1] * imgHeight, bb[1]* imgHeight)
                    ixmax = np.minimum(BBGT[:, 2] * imgWidth, bb[2]* imgWidth)
                    iymax = np.minimum(BBGT[:, 3] * imgHeight, bb[3]* imgHeight)
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih
                    uni = ((bb[2] - bb[0]) * imgWidth * (bb[3] - bb[1]) * imgHeight +
                           ((BBGT[:, 2] - BBGT[:, 0]) * imgWidth) *
                           ((BBGT[:, 3] - BBGT[:, 1]) * imgHeight) - inters)  # hyk modified , fix  bug here
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > overlap_thresh:
                    if BBdet[jmax]:
                        nd = nd - 1
                    else:
                        BBdet[jmax] = 1
                        tp[ind] = 1.
                        ind += 1
                else:
                    fp[ind] = 1.
                    ind += 1
            # compute precision recall
            fp = fp[0:nd]
            tp = tp[0:nd]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult

            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self.class_ap(rec, prec, use_07_metric)
            #print('postive samples number: ', npos)
            #print('false postive number:', fp)
            #print('true postive number:', tp)
        else:
            #print(detectionPath, 'empty!')
            rec = 0.
            prec = 0.
            ap = 0.

        return rec, prec, ap


    def reformat_gt(self, gts):
        """
        reformat ground truth data.
        :param gts: {'image_name':
                        { 'instances':
                            [{'type': str, 'bbox': [x0, y0, x1, y1]},
                            {'type': str, 'bbox': [x0, y0, x1, y1]},
                            ...]
                        },
                     'image_name2': {...},
                     ...,
                    }
        :return: None
        """
        for imgname, annots in gts.items():
            annot = []
            for instance in annots['instances']:
                annot.append([instance['type']] + instance['bbox'][:4])

            self.imgname2GT[imgname] = annot

    def reformat_result(self, det_results):
        """
        reformat ground truth data.
        :param det_results: {'image_name':
                                [{'type': str, 'bbox': [x0, y0, x1, y1]},
                                {'type': str, 'bbox': [x0, y0, x1, y1]},
                                ...],
                             'image_name2':
                                [...],
                             ...,
                            }
        :return: None
        """
        for imgname, results in det_results.items():
            for result in results:
                # x: [imgname, score, xmin, ymin, xmax, ymax]
                try:
                    self.detect_results[result['type']].append([imgname, result['confidence']] + result['bbox'][:4])
                except:
                    self.detect_results[result['type']] = [[imgname, result['confidence']] + result['bbox'][:4]]

    def calc_map(self, gts, det_results):
        self.reformat_gt(gts)
        self.reformat_result(det_results)
        MAP, recall, prec = self.do_python_eval()
        return MAP,recall, prec
    
# def calculate_map(gts, detect_results):
#     calc_map = CalcMAP(['hand'])
#     final_map = calc_map.calc_map(gts, detect_results)
#     return final_map


# if __name__ == '__main__':
#     calc_map = CalcMAP(['hand'])
#     gts = { 'image1': {'instances':
#                       [{'type': 'hand', 'bbox': [0, 0, 0.5, 0.5, 1]}]
#                     },
#             'image2': {'instances':
#                       [{'type': 'hand', 'bbox': [0, 0, 0.5, 0.5, 1]}]
#                     }
#             }
#     detect_results = {
#         'image1': [
#             {'type': 'hand', 'bbox': [0, 0, 0.4, 0.4], 'confidence': 1},
#         ],
#         'image2': [
#             {'type': 'hand', 'bbox': [0.4, 0.4, 1, 1], 'confidence': 0.9},
#             {'type': 'hand', 'bbox': [0, 0, 0.5, 0.5], 'confidence': 0.8}
#         ]

#     }
#     print(calc_map.calc_map(gts, detect_results))
