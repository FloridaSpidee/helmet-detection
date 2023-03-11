# YOLO模型
>
## Explanation
### `demo.py`
  单独使用线程thread1来进行检测       
  `def detect(self,imgs):`用于接收一个图片列表（因为没有做batch，但为了统一接口只能传入[img]的形式）

>大致思路  

设置`interval`的值，本次设为3，以三帧为一个小节。第一个小节不显示。

第一帧需要检测，把图片传入线程`thread1`，开始检测，把结果保存在`self.out`中，同时读入第二和第三帧存入`Queue`实例`img_que`。

第二小节开始展示。在第二个小节开始时，即读到第四帧时，从`thread1`中取出第一帧的图片，和结果，同时重新传入第四帧进行检测，更新`mot_tracker`，画框展示。第五第六帧都是传入`img_que`并`img=img_que.get()`取出三帧前的图片。`mot_tracker`预测框，画框并展示。

后面类推

### `demo_only_detect.py`
单独只使用Yolo检测器，无tracker隔帧和多线程检测。

### `detection_module_withdelete.py`
有去除误检功能，文件名去掉'_withdelete'替换掉`detection_module.py`文件即可。