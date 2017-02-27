#Draft: Deep Learning in Neural Networks: An Overview  
origin:http://people.idsia.ch/~juergen/DeepLearning15May2014.pdf  
origin-translate:https://github.com/CACppuccino/Machine-Learning-Notes/blob/master/Draft:%20Deep%20Learning%20in%20Neural%20Networks:%20An%20Overview.md or 
##注：由于译者知识有限，本篇可能需要继续学习才能将其正确地译出，所以滚去学习了555  
####作者：Jurgen Schmidhuber  
####翻译：Lord Cup  
###不确定的terms  
Description length-描述长度
Learining Hierarchical Representation-学习分层代表  
Group Method of Data Handling(GMDH)-分组法处理数据  
Winner-Take-All(WTA)-赢者通吃  
##摘要  
近年来，深度人工神经网络（包含递归神经网络）领域涌现出了大量的关于特征识别和机器学习的文章。本文简洁地总结了大部分有关20世纪的成果。按照在该领域所作出的贡献大小，即动作与影响之间的链接，来区分浅层与深层学习者。我回顾了深度监督学习（同时概况了BP算法的历史）、无监督学习、加强学习和进化计算，还有间接的关于简短编写的深层和大型网络的程序的调查。  
##前言  
这是一个关于深度学习（Deep Learning(DL)）的概述的草稿。目的之一是赞扬那些使如今的机器学习达到极高层次的杰出贡献者们。我承认目前至想达到这个目标还有一些距离。DL研究社区自身可能被看作持续参与的，深度网络化的一群科学家，他们以一种复杂的方式互相影响着。以近期的DL成果为起点，我尝试回溯过去半个多世纪相关的思想的来源，有时采用“本地搜索”，追踪文章中的引用来回到过去。由于并非所有的DL文献书籍会承认之前的相关成果，额外的全局搜索也被应用在调查之中，同时我们也借助了许多神经网络专家的帮助。结果是，当今的草稿中包含了大量的参考（至今为止大约800份）。然而，我可能会由于本身的偏见而忽视了一些重要的成果。正如其中一种偏见就起源于我对自己DL成果的过度熟悉。由于这些原因，当前的草稿应当被看做仅仅是对正在进行的荣誉认可过程的概述。欢迎各位置信我们，来帮助我们改善这篇作品。联系邮箱：juregen@idsia.ch  
##内容  
### 1 对深度学习（DL）在神经网络（NNs）中的介绍  
### 2 Event-Oriented Notation for Activation Spreading in FNNs/RNNs  
### 3 Depth of Credit Assignment Paths (CAPs) and of Problems  
### 4 深度学习中的循环  
#### 4.1 DL的动态编程（DP）  
#### 4.2 非监督学习（UL）对监督学习（SL）和RL的促进  
#### 4.3 Occam剃刀：对描述长度的压缩与缩小(MDL)  
#### 4.4 SL,UL,RL的学习分层代表  
#### 4.5 神经网络的深度学习中图像单元的快速处理  
### 神经网络的监督学习，其中一些受助于非监督神经网络  
#### 5.1 1940年代及更早时期  
#### 5.2 1960左右：深度学习从生物神经学得到更多启发  
#### 5.3 1965：基于分组法处理数据（GMDH）的深度网络  
#### 5.4 1979：卷积+权重复制+赢者通吃(WTA)  
#### 5.5 1960-1981和之后：神经网络中的BP算法的发展  
##### 5.5.1 BP在前馈神经网络和递归神经网络的权重分享
