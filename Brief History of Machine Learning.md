#Brief History of Machine Learning#

##origin:http://www.erogol.com/brief-history-machine-learning/  
###翻译：Cup  
###一些有可能翻译不准确的terms  
Hebbian Learning theory
Recurrent Neural Network-递归神经网络    
Delta Learning rule-三角洲学习规则  
Least Square problem-最小平方问题  
Perceptron-感知器  
convex optimization-凸优化  
generalization margin theory-广义边际理论  
gradient loss-梯度损失
##正文  
机器学习时间轴：http://www.erogol.com/wp-content/uploads/2014/05/test.jpg  
  
  由于科学之最初立场，科技与人工智能，科学家们跟随着Blaise Pascal和Von Leibniz的脚步思考是否有一种机器，拥有与人类相同的智能。著名作者如Jules Verne，Frank Baum(绿野仙踪)，Marry Shelly(弗兰肯斯坦)，George Lucas(星球大战)幻想着人造人有着与人类相似甚至更强的能力，在不同情况下拥有着人性化的能力。  
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
  一个关于能够不明确编程即使计算机拥有某种能力的研究领域。  
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
  之前，另一个名为Adaboost机器学习模型被Freund和Schapire于1997年提出，描述为弱分类器的增强组合。这份论文在当时为其作者带来了哥德尔奖(Godel Prize)。Adaboost通过提供重要的
