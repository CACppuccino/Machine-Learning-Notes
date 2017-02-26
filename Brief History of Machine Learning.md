#Brief History of Machine Learning#

##origin:http://www.erogol.com/brief-history-machine-learning/  
####翻译：Lord Cup  
####特别感谢：Song
###一些有可能翻译不准确的terms  
Hebbian Learning theory
Recurrent Neural Network-递归神经网络    
Delta Learning rule-三角洲学习规则  
Least Square problem-最小平方问题  
Perceptron-感知器  
convex optimization-凸优化  
generalization margin theory-广义边际理论  
gradient loss-梯度损失  
decision-theoretic-理论决策  
exogenous parameters-外源参数 
##译者前言  
回顾历史，能够让我们更加清晰地看清ML的去向。更重要的是，AI的历史无数次证明，没有一个流派会永远辉煌，也没有一个流派会一直沉寂。  
##正文  
机器学习时间轴：http://www.erogol.com/wp-content/uploads/2014/05/test.jpg  
  
  由于科学、技术与人工智能之最初立场，科学家们跟随着Blaise Pascal和Von Leibniz的脚步，思考是否有一种机器，拥有与人类相同的智能。著名作者如Jules Verne，Frank Baum(绿野仙踪)，Marry Shelly(弗兰肯斯坦)，George Lucas(星球大战)幻想着人造人有着与人类相似甚至更强的能力，在不同情况下拥有着人性化的能力。  
Pascal的机器表演加减法：http://www.erogol.com/wp-content/uploads/2014/05/Arts_et_Metiers_Pascaline_dsc03869.jpg  
  机器学习(Machine Learning)是人工智能（AI）非常热门的一个方向，无论是在学术界还是工业界。公司，大学投入了大量的资源来提升他们在这方面的知识。近期的在此领域的进步，在很多任务中取得了实实在在的成果。  
  在这里我将分享一个机器学习的时间轴，并标注出其中的一些里程碑（尽管不一定完全）。  
  向着机器学习前进的第一步是由Hebb在1949年迈出的，基于一个神经心理学的学习公式。它被称为Hebbian学习原理。简单来讲，它加紧了节点和递归神经网络的相关性。
  它记住了所有网络上的共性并在之后如同人的记忆一般工作。正式地来讲，关于它的论述如下：  
  **让我们假设一个反射活动的持久性或重复性倾向于引诱持续的细胞发生改变从而增加它的稳定性。...
  当一个神经细胞A的轴突足够近来使细胞B兴奋并重复或持续地参与激活它，一些增长的过程或新陈代谢的改变会在其中一个或双方细胞中发生。
  这样的话，A的效率在刺激B的同时会升高。[1]**  
  在1952年，Arthur Samuel在IBM研发出了会下跳棋的程序。该程序能够观察位置并学习得出一个
  模糊的模型，指导其为以后的情况作出更好的移动。Samuel与程序下了很多次棋，并发现程序能够随着时间的增长而下的更好。  
  Samuel借此反驳了公众关于机器执行的命令不能够超出所写代码并如人类一般学习的认知。他创造了“机器学习(machine learning)”这一术语，并定义：  
  **一个通过不明确编程就使计算机拥有某种能力的研究领域。**  
  1957年，Rosenblatt的Perceptron是第二个以神经科学为背景提出的模型，并与当今的机器学习模型
  相似。这在当时，是一个非常激动人心的发现，并比Hebbian的理念更加具有适用性。
  Rosenblatt是这样介绍感知器Perceptron的：  
  **Perceptron是用于展示一些总体上的智能系统的基本特性而设计的，不是针对
  个别生物所拥有的太特别和未知的情况。[2]**  
  3年之后，Widrow[4]制定了三角洲学习规则用于实际中训练感知器(Perceptron)的过程。这也被称为最小平方问题。两个理念的结合
  产生了一个不错的线性分类器。然而，Percptron带来的兴奋因Minsky[3]的出现而在1969年戛然而止。他提出了著名的亦或问题和Perceptrons在该线性不可分割数据分布的无力。正是Minsky使神经网络社区的发展停滞。之后，神经网络的研究者们的活动一直处于休眠状态知道20世纪80年代。  
  亦或图：http://www.cs.ru.nl/~ths/rt2/col/h10/draw-LTUdecis.GIF  
  在1981年之前，社区一直未再作出更多的前进，直到Werbos[6]通过神经网络下的BP算法提出多层感知器（MLP），尽管BP的概念被Linnainmaa[5]于1970年以“逆向自动区分模型”提出过。BP算法仍然是如今神经网络架构中重要的成分之一。有这些新思想的提出，神经网络的研究又一次加速。在1985-1986年神经网络的研究者们成功地展示了以BP训练出的MLP(Rumelhart,Hinton,Willianms[7]-Hetch,Nielsen[8])。  
  来自Hetch和Nielsen：http://www.erogol.com/wp-content/uploads/2014/05/Hetch_Nielsen_NN.png
  另一方面，决策树这一著名的机器学习算法被J.R.Quinlan[9]于1986年提出，详细来说是ID3算法。这是机器学习另一主流闪现灵。火花的重要象征点。更重要的是，ID3被实现为一个能够解决更多实际生活应用问题的软件，基于其简洁的规则和清晰的推理，与黑盒下的神经网络模型恰恰相反。  
  在ID3之后，更多的改变与提升涌现出来（如ID4，回归树，分类与回归树（CART）等等）并且在至今的机器学习中仍然是活跃的话题之一。  
  一个简单决策树（来源Quinlan[9]）：http://www.erogol.com/wp-content/uploads/2014/05/Quinlan_ID3.png  
  机器学习的一大重要突破来源于支持向量机（SVM），由Vapnik和Cortes[10]于1995年在非常有力的理论依据和经验结果下得出。正是这时，机器学习社区分为了神经网络（NN）和支持向量机（SVM）两大拥护群体。然而，对于神经网络的支持者来说，在Kernelized于近2000年提出新版本SVM后，竞争的优势并不在他们这里。SVM在原来许多神经网络成功的案例中取得了更好的成绩。同时，SVM能够利用所有之前凸优化，广义边际理论和核心来对抗NN模型。因此，它能够从不同的领域得到巨大的理论和实践的发展。  
  二维分类问题（来源Vapnik和Cortes[10]）http://www.erogol.com/wp-content/uploads/2014/05/SVM_Vapnik.png  
  NN又一次遭受重击是源于Hochreiter的于1991的论文[40]和Hochreiter与其他人合作于2001的论文[11]，说明了NN的神经单元在应用了BP学习之后饱和带来的梯度损失。简单来讲，NN神经元由于饱和会在训练一段时间后出现冗余，因此NN倾向于在较短的训练时期内出现过度拟合问题。  
  之前，另一个名为Adaboost机器学习模型被Freund和Schapire于1997年提出，描述为弱分类器的增强组合。这份论文在当时为其作者带来了哥德尔奖(Godel Prize)。Adaboost通过提供更多重要的硬性实例来训练一套易于训练的弱分类器。该模型至今仍然是如脸部识别和侦察的基础。它同时也是PAC（Probably Approximately Correct）学习理论的一种实现。总体来讲，所谓的弱分类器是被选择为简单的决定步骤（单个决策树节点）。它们如此介绍Adaboost：  
  **我们所学习的这个模型，可以被看做是一个线上善于学习的预测模型，至一个总体的理论决策设置的更广泛、抽象的拓展...[11]**  
  另一个于2001年被Breiman[12]提出的合成模型，能够合成多个决策树，而每个决策树是由实例的一套随机子集决策，并且每个节点是通过参数的一个随机子集产生的。由于它本身的特性，它被称为随机森林(Random Forests,RF)。RF也有理论和实际经验来证明对过度拟合的耐性。即使是AdaBoost也显示出了对于处理过度拟合的上的短板和对数据中异常实例处理的无力，而RF对这些问题有着更强大的处理能力。RF在许多不同的任务中取得了成功，在Kaggle的比赛中也是如此。（Kaggle网站：https://www.kaggle.com/competitions 译者注）。
  **随机森林（RF）一组决策树的组合，每一棵树依赖于一个独立抽样得到的随机向量的值，并且森林中的每个树的分布是相同的。森林收敛的通常错误如在树的数目增加时会达到极限。[12]**  
  当时间逐渐来到今天时，一个神经网络的新时代——深度学习——来临了。该术语指参考NN神经网络模型并使用大量连续的神经层。神经网络的第三次大概崛起于2005年，借助于从过去到现在的不同发现而被Hinton,LeCun,Bengio,Andrew Ng和其他可贵的研究者们实现除了。以下列出了部分重要的标题：  
  
GPU programming  
Convolutional NNs [18][20][40]  
Deconvolutional Networks [21]  
Optimization algorithms  
Stochastic Gradient Descent [19][22]  
BFGS and L-BFGS [23]  
Conjugate Gradient Descent [24]  
Backpropagation [40][19]  
Rectifier Units  
Sparsity [15][16]  
Dropout Nets [26]  
Maxout Nets  [25]  
Unsupervised NN models [14]  
Deep Belief Networks [13]  
Stacked Auto-Encoders [16][39]  
Denoising NN models [17]  在
  在这些（和一些没有罗列在这里）的思想的结合下，神经网络模型拥有了完成许多任务的能力，如物体识别，演讲识别，自然语言处理等。然而，要强调的是，这并不代表着其他机器学习流派的终结。即使深度学习的成功例子在快速增加，仍有着很多的批评指向模型的训练代价和对外源参数的调整能力。更进一步，SVM仍然由于其简易性而被广泛运用。（这样说可能会引起巨大的争议:)-作者注）  
  -在结束之前，我要触及另一个年轻的机器学习分支。在万维网和社交媒体的增长下，一个新术语-大数据-产生了并对机器学习有着巨大的影响。由于大数据带来的巨大问题，许多强力的机器学习算法由于某些原因而变得无力（当然对大型科技公司不是问题）。因此，研究者们研发出了一套新的模型，被称为Bandit算法[27-38]（用于线上学习）使学习更简单，更能够适应大数据规模问题。  
  若本文有任何问题，欢迎联系我并及时指出。  
  ---------------------------------------  
  References ----
[1] Hebb D. O., The organization of behaviour.New York: Wiley & Sons.

[2]Rosenblatt, Frank. "The perceptron: a probabilistic model for information storage and organization in the brain." Psychological review 65.6 (1958): 386.

[3]Minsky, Marvin, and Papert Seymour. "Perceptrons." (1969).

[4]Widrow, Hoff "Adaptive switching circuits." (1960): 96-104.

[5]S. Linnainmaa. The representation of the cumulative rounding error of an algorithm as a Taylor
expansion of the local rounding errors. Master’s thesis, Univ. Helsinki, 1970.

[6] P. J. Werbos. Applications of advances in nonlinear sensitivity analysis. In Proceedings of the 10th
IFIP Conference, 31.8 - 4.9, NYC, pages 762–770, 1981.

[7] Rumelhart, David E., Geoffrey E. Hinton, and Ronald J. Williams. Learning internal representations by error propagation. No. ICS-8506. CALIFORNIA UNIV SAN DIEGO LA JOLLA INST FOR COGNITIVE SCIENCE, 1985.

[8] Hecht-Nielsen, Robert. "Theory of the backpropagation neural network." Neural Networks, 1989. IJCNN., International Joint Conference on. IEEE, 1989.

[9] Quinlan, J. Ross. "Induction of decision trees." Machine learning 1.1 (1986): 81-106.

[10] Cortes, Corinna, and Vladimir Vapnik. "Support-vector networks." Machine learning 20.3 (1995): 273-297.

[11] Freund, Yoav, Robert Schapire, and N. Abe. "A short introduction to boosting."Journal-Japanese Society For Artificial Intelligence 14.771-780 (1999): 1612.

[12] Breiman, Leo. "Random forests." Machine learning 45.1 (2001): 5-32.

[13] Hinton, Geoffrey E., Simon Osindero, and Yee-Whye Teh. "A fast learning algorithm for deep belief nets." Neural computation 18.7 (2006): 1527-1554.

[14] Bengio, Lamblin, Popovici, Larochelle, "Greedy Layer-Wise
Training of Deep Networks", NIPS’2006

[15] Ranzato, Poultney, Chopra, LeCun " Efficient Learning of  Sparse Representations with an Energy-Based Model ", NIPS’2006

[16] Olshausen B a, Field DJ. Sparse coding with an overcomplete basis set: a strategy employed by V1? Vision Res. 1997;37(23):3311–25. Available at: http://www.ncbi.nlm.nih.gov/pubmed/9425546.

[17] Vincent, H. Larochelle Y. Bengio and P.A. Manzagol, Extracting and Composing Robust Features with Denoising Autoencoders, Proceedings of the Twenty-fifth International Conference on Machine Learning (ICML‘08), pages 1096 - 1103, ACM, 2008.

[18] Fukushima, K. (1980). Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position. Biological Cybernetics, 36, 193–202.

[19] LeCun, Yann, et al. "Gradient-based learning applied to document recognition."Proceedings of the IEEE 86.11 (1998): 2278-2324.

[20] LeCun, Yann, and Yoshua Bengio. "Convolutional networks for images, speech, and time series." The handbook of brain theory and neural networks3361 (1995).

[21] Zeiler, Matthew D., et al. "Deconvolutional networks." Computer Vision and Pattern Recognition (CVPR), 2010 IEEE Conference on. IEEE, 2010.

[22] S. Vishwanathan, N. Schraudolph, M. Schmidt, and K. Mur- phy. Accelerated training of conditional random fields with stochastic meta-descent. In International Conference on Ma- chine Learning (ICML ’06), 2006.

[23] Nocedal, J. (1980). ”Updating Quasi-Newton Matrices with Limited Storage.” Mathematics of Computation 35 (151): 773782. doi:10.1090/S0025-5718-1980-0572855-

[24] S. Yun and K.-C. Toh, “A coordinate gradient descent method for l1- regularized convex minimization,” Computational Optimizations and Applications, vol. 48, no. 2, pp. 273–307, 2011.

[25] Goodfellow I, Warde-Farley D. Maxout networks. arXiv Prepr arXiv …. 2013. Available at: http://arxiv.org/abs/1302.4389. Accessed March 20, 2014.

[26] Wan L, Zeiler M. Regularization of neural networks using dropconnect. Proc …. 2013;(1). Available at: http://machinelearning.wustl.edu/mlpapers/papers/icml2013_wan13. Accessed March 13, 2014.

[27] Alekh Agarwal, Olivier Chapelle, Miroslav Dudik, John Langford, A Reliable Effective Terascale Linear Learning System, 2011

[28] M. Hoffman, D. Blei, F. Bach, Online Learning for Latent Dirichlet Allocation, in Neural Information Processing Systems (NIPS) 2010.

[29] Alina Beygelzimer, Daniel Hsu, John Langford, and Tong Zhang Agnostic Active Learning Without Constraints NIPS 2010.

[30] John Duchi, Elad Hazan, and Yoram Singer, Adaptive Subgradient Methods for Online Learning and Stochastic Optimization, JMLR 2011 & COLT 2010.

[31] H. Brendan McMahan, Matthew Streeter, Adaptive Bound Optimization for Online Convex Optimization, COLT 2010.

[32] Nikos Karampatziakis and John Langford, Importance Weight Aware Gradient Updates UAI 2010.

[33] Kilian Weinberger, Anirban Dasgupta, John Langford, Alex Smola, Josh Attenberg, Feature Hashing for Large Scale Multitask Learning, ICML 2009.

[34] Qinfeng Shi, James Petterson, Gideon Dror, John Langford, Alex Smola, and SVN Vishwanathan, Hash Kernels for Structured Data, AISTAT 2009.

[35] John Langford, Lihong Li, and Tong Zhang, Sparse Online Learning via Truncated Gradient, NIPS 2008.

[36] Leon Bottou, Stochastic Gradient Descent, 2007.

[37] Avrim Blum, Adam Kalai, and John Langford Beating the Holdout: Bounds for KFold and Progressive Cross-Validation. COLT99 pages 203-208.

[38] Nocedal, J. (1980). "Updating Quasi-Newton Matrices with Limited Storage". Mathematics of Computation 35: 773–782.

[39] D. H. Ballard. Modular learning in neural networks. In AAAI, pages 279–284, 1987.

[40] S. Hochreiter. Untersuchungen zu dynamischen neuronalen Netzen. Diploma thesis, Institut f ̈ur In-
formatik, Lehrstuhl Prof. Brauer, Technische Universit ̈at M ̈unchen, 1991. Advisor: J. Schmidhuber.
