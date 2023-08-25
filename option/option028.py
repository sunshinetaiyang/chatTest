
# option028【择时06】“过度偏误”策略
# =======================
# 策略简介2019-02-25 
# 50ETF涨幅>3%的情形非常少，这样大的涨幅很多是属于情绪过于热烈，一般有反转潜力
# 聚宽的期权数据只能从170101开始，通过对20170101至20190228的数据回测，开仓10次，胜率80%
# 胜率还是非常优秀的，说明了这个策略的有效性，值得深入研究
# 加上通过止盈止损的差别设定，控制盈亏比，模型的盈利可观，2年5倍，最大回撤-15%

# 策略优化2019-06-25
# 2017年全年没有3%振幅以上，2018年11单，结果2019年6月还没结束已经11单了，今年的振幅很大很频繁。
# 今天非常痛心，11号开仓LP由于20号大涨止损退出，结果就怀疑模型，没有继续开仓LP，结果很快就大跌，LP获利没有抓住。
# 亏损你都在，赚钱不在，让人非常的不爽，而且20号那天是挂单的，差1个基点没有成交。如果按照模型开仓，今天已经把上次亏损扳回。
# 痛心的是为什么不按照模型来，结果模型一次次证明其正确性。不多说，还是要严格按照模型，改进模型。
# 目前的几个问题。1，止损幅度问题；2，方向错掉之后的后招；3，是否使用卖方替换买方？4，是否加倍下注问题。
# 第一个问题，止损幅度，因为行情一直还是有变动，2018年的数据其参考权重不能给那么高，尤其是一个重要的止损系数
#             止损以2019年行情数据为主，另外要参考止盈，盈利比要大于1，止损不能高于40%。
# 第二个问题，方向错掉之后，还是等止损，不做其他动作。
# 第三个问题，卖方替代问题，马上开发新模型代码进行回测。
# 第四个问题，目前暂不加倍，但是止损剩下的资金，加上1万一起买入，实际上是增大投入。
# =======================

# =======================
# 版本更新记录
# 2019-02-09 文件创建
# 2019-05-06 改进档位选择逻辑，抽象为函数，定义全局变量统一规范控制
# 2019-05-07 strBuyTimebyMin  = ' 14:58:00' 入场买入时间推迟到14:58，从0506的实盘和2019六次回测来看可提升10%收益率
# 2019-05-07 TrailingCheck 封装止盈止损函数，逻辑上加大止损止损判断密度，尝试更高收益率
# 2019-05-10 回测取实一档时，2月25号大涨太猛，以至于认沽的实一档还没挂出合约，开仓失败。修改kaicang_LP函数，失败时取实值。
# 2019-05-11 已有CO/PO头寸情况下，标的和期望反向运行3%，之前是移仓操作，移仓后WinStop退出，导致打成平手，修改为固守策略。
# 2019-05-12 回测远月合约LeftDay=30时，由于时限选不出合约，增加query时的时间选择跨度到150天
# 2019-06-12 期权市场甚至是整个金融市场还是会因为宏观因素变化导致策略有不同的适应性。对028模型来说，2018年远月实一档胜率很高。
#            但是2019年至今，胜率很一般，为了研究更实际可用的模型，专注2019年从底部起来之后的特征，开仓调到>2.6%，各类参数
#            也根据回测情况进行修正，看看是否让028模型可以带来更多的收益。
# 2019-06-12 dfContext新增开仓触发涨幅数据，以及头寸持有天数。
# 2019-06-13 修改止盈逻辑，以标的幅度为判断依据，避免波动率影响。居然只影响了20190510这一单，明显是波动率下降导致。
#            if fltCurPoPrice  > fltCostPrice*WinStop or flt50ETFCur < flt50ETFPrice*0.97: 
# 2019-06-25 策略优化，详情如上面注释
# 2020-08-22 时隔一年多，经历了2019年下半年的“上升态势中持续卖出实值认沽”的卖方策略，以及2020年大疫情影响的大跌，重新思考期权的本质。
#            期权本来是价格的“保险”是把“不确定”变成“确定”的工具，要做方向不是不可以而是难度极大。综合巴菲特和老唐关于价格的理论，
#            价格只能“被利用”而不是可预测的，“如何利用价格”才是“做价格”的本质，而价格能做只有在“时机驱动”，在“过度偏误”，
#            只有“过度偏误”的价格才可能保持高的胜率和盈亏比。整个期权除了做保险，要做方向，只有“过度偏误”才是正途。
#            再次深入来研究“过度偏误”发生的规律和把握方法，做好这个体系。
# 2020-08-22 review去年代码，增加注释，按照新的思路做必要修改。
# 2020-09-28 暂缓。模型改进，修改为第二天早盘开仓，但是由于joinquant无法跟踪实时期权数据，开仓信号需要另外想办法进行监控，要发送明确信号！！
# 2020-09-28 增加合约到期时平仓逻辑。回测起点datetime.date(2019, 6, 1)时，发生未平仓逻辑，原因是既没有触发止盈又没触发止损，
#            但模型又没有到期平仓的逻辑。到期不平仓肯定不对，028模型是过激反转逻辑，最少给了20天时间持有头寸，20天都没有发生反转，“过度”已经被时间消化，模型中也有
#            监控头寸持有时间的设定参数dfContext.days，只有一单2019.6.20这一单是超时未触发止盈止损的。
# 2020-09-29 增加对300ETF的回测，新增前再review一下代码，规范几个交易函数名字->pingcang_LC等
# 2020-10-16 10月11号实盘10万开仓，这几天不温不火，还是像仔细看看历史上开单后各头寸的走势情况，以便这次可以拿得稳些。
#            增加函数anynisis_positions()，传入交易列表df，打印出各次开单之后头寸的变现。
# 2020-11-09 再次review代码，改进df处理的相关逻辑，并把结果记录于Excel
# =======================

# =======================
# 模型关键参数2019-05-06
# 该取值有以下4个。目前取1.4止盈，0.6追踪止损，InOutLevel虚一档，>20天行权
# WinStop       = 1.4  # 止盈平仓
# LossStop      = 0.6  # 追踪止损闪人平仓
# InOutLevel    = -1   # 虚实度设定。0为平值，-1为虚一档，1为实一档，以此类推
# LeftDays      = 20   # 用于选择行权日间隔，近远月合约时间选择
# opt.OPT_DAILY_PREOPEN.exercise_date>dateHandle+datetime.timedelta(days=LeftDays)
# =======================
# 好模型要专注精深。止盈止损点可以优化。剩余时间和虚实度更是有规律可言，需要进行优化。
# =======================
# 2019-05-06特朗普Twitter发布贸易战继续加关税的消息，50在中午收盘跌5%，触发开仓条件
# 但问题是期权的价格还是高，是否有改进方案，比如买入更虚值？或者用义务方来实现模型效果？
# 调参1：买入虚2档。关于买入虚值，虚多少，其实和买入日期非常相关，如果到行权日还长，虚值会有优势，优势被放大
# 反之则需要买入较为实值甚至平值
# =======================

# =======================
# 模型关键参数2019-05-13
# 经过一周的回测和代码优化，取值如下，具体记录在如下参数回测注释中。
# WinStop       = 1.4  # 止盈平仓
# LossStop      = 0.4  # 追踪止损闪人平仓，该取值为配合“固守”替代“移仓”
# InOutLevel    = 1   # 虚实度设定。0为平值，-1为虚一档，1为实一档，以此类推
# LeftDays      = 20   # 用于选择行权日间隔，近远月合约时间选择

# =======================
# 模型参数分析01。LeftDays取值。主要在当月和下月的选取。
# 2019-05-10  结论，取LeftDays = 20，下月选择容错率更高，虽然牺牲了收益，适合当前，总需要赢一笔吧。
# WinStop       = 1.4  # 止盈平仓
# LossStop      = 0.6  # 追踪止损闪人平仓
# InOutLevel    = -1   # 虚实度设定。0为平值，-1为虚一档，1为实一档，以此类推
# ==LeftDays      = 20 区间2018.1.1-2019.5.10 最高值 516,543；期末值 488,827；最大回撤 -5.37%。
# ==LeftDays      = 10 区间2018.1.1-2019.5.10 最高值 638,603；期末值 581,833；最大回撤 -8.89%。
# ==LeftDays      = 30 区间2018.1.1-2019.5.10 最高值 393,204；期末值 375,202；最大回撤 -4.57%。

# 2019-05-12 回测在两次连续大跌情况下，不做移仓只做固守的模型表现。
# WinStop       = 1.4  # 止盈平仓
# LossStop      = 0.4  # 为了配合固守逻辑，让止损更宽容0.6->0.4
# InOutLevel    = 1   # 虚实度设定。0为平值，-1为虚一档，1为实一档，以此类推
# ==LeftDays      = 10 区间2018.1.1-2019.5.10 最高值 502846；期末值 466190；最大回撤 -61.89% 胜率10/13 近月跌幅大导致一笔止损
# ==LeftDays      = 20 区间2018.1.1-2019.5.10 最高值 561796；期末值 535693；最大回撤 -61.89% 胜率11/13 高胜率高收益率
# ==LeftDays      = 30 区间2018.1.1-2019.5.10 最高值 432742；期末值 353362；最大回撤 -68.51% 胜率9/11 远月合约涨幅不够
# 2019-05-12 观察各期合约的规律，LeftDays值是否可以更智能，可变，包括虚实度，能否更为智能。
# =======================

# =======================
# 模型参数分析02。InOutLevel取值，虚实度，哪个虚实度在这个模型有更好的表现。
# 2019-05-10  结论：真是没想到，太颠覆世界观了。以前老是认为虚值涨得快涨得猛，这一回测算是明白了，没有无缘无故的爱。
#                   虚值涨得快，也同样跌得快，暴涨暴跌毕竟少见，否则这个模型都是买入后1天都止盈退出，
#                   现实是极少买入后立即能够达到止盈的，说明什么，都是时间熬出来的，既然没有急涨急跌，那就是实值更稳了。
#             结论：居然是取实1档最好。在胜率、最高值、回撤的表现，综合最佳
# 保持其他3个关键参数固定
# WinStop       = 1.4  # 止盈平仓
# LossStop      = 0.6  # 追踪止损闪人平仓
# LeftDays      = 20   # 用于选择行权日间隔，近远月合约时间选择 
# ==InOutLevel    = -2。区间2018.1.1-2019.5.10 最高值 464,908；期末值 379,961；最大回撤 -51.16%，胜率 9/15
# ==InOutLevel    = -1。区间2018.1.1-2019.5.10 最高值 516,543；期末值 488,827；最大回撤 -50.45%，胜率 10/15
# ==InOutLevel    =  0。区间2018.1.1-2019.5.10 最高值 512,551；期末值 491,327；最大回撤 -47.13%，胜率 11/15
# ==InOutLevel    =  1。区间2018.1.1-2019.5.10 最高值 522,260；期末值 513,647；最大回撤 -44.22%，胜率 11/15 The Best!
# ==InOutLevel    =  2。区间2018.1.1-2019.5.10 最高值 503,727；期末值 493,992；最大回撤 -40.88%，胜率 11/15
# ==InOutLevel    =  3。区间2018.1.1-2019.5.10 最高值 509,366；期末值 487,754；最大回撤 -56.67%，胜率 11/15
# =======================

# =======================
# 模型参数分析03。WinStop LossStop 止盈止损参数。
# WinStop       = 1.4  # 止盈平仓
# LossStop      = 0.6  # 追踪止损闪人平仓
# 这两个参数最核心的任务就是提高胜率，提高整体的胜率，提升了胜率，总收益率自然上升。
# 2019-02-09 做过以下回测
# 0.5,0.6,0.7三个值都取到了，0.6最大，意思是0.7给出的容错太小影响了后面的上涨，
# 0.6往下基本上就是没啥指望了，不会再涨起来，可以止损。
#
# 2019-05-11 回测分析WinStop取值。保持其他3个参数值不变
# 结论：WinStop    =  1.4取值最佳，太低则盈亏比低而且胜率没有进化，再高则胜率下降，目前没盈利，还是希望胜率高。取1.4
# InOutLevel    = 1
# LossStop      = 0.6  # 追踪止损闪人平仓
# LeftDays      = 20   # 用于选择行权日间隔，近远月合约时间选择 
# ==WinStop    =  1.2 区间2018.1.1-2019.5.10 最高值 341632；期末值 306152；最大回撤 -44.22%，胜率 11/15 盈亏比低
# ==WinStop    =  1.3 区间2018.1.1-2019.5.10 最高值 482660；期末值 463796；最大回撤 -44.22%，胜率 11/15 盈亏比低
# ==WinStop    =  1.4 区间2018.1.1-2019.5.10 最高值 522260；期末值 513647；最大回撤 -44.22%，胜率 11/15 The Best
# ==WinStop    =  1.5 区间2018.1.1-2019.5.10 最高值 552239；期末值 546568；最大回撤 -44.22%，胜率 10/15 胜率下降
# ==WinStop    =  1.6 区间2018.1.1-2019.5.10 最高值 440359；期末值 420842；最大回撤 -44.22%，胜率  8/15 胜率太低
# ==WinStop    =  1.7 区间2018.1.1-2019.5.10 最高值 508986；期末值 498066；最大回撤 -44.22%，胜率  8/15 胜率太低
# =======================

# =======================
# 模型参数分析 2019-05-11 总结：经过系统的对4个参数进行回测比较，确定如下取值为目前最优。后续还会依据实盘进行优化。
# WinStop       = 1.4  # 止盈平仓
# LossStop      = 0.6  # 追踪止损闪人平仓
# InOutLevel    = 1   # 虚实度设定。0为平值，-1为虚一档，1为实一档，以此类推
# LeftDays      = 20   # 用于选择行权日间隔，近远月合约时间选择 
# =======================

# =======================
# 回测分析
# 2019-02-27 回测，逐笔观察028模型下每次交易的具体情况
# 2019-05-11 回测，改进移仓逻辑，具体如下：
# 2019-05-11 止损单分析。共4条，其中有以下2条，都是移仓时的止损，可以看到后续都止盈出场，所以完全可以不移仓。
# No.1 买入认购后继续大跌，移仓到更低的认购，止损加止盈，基本上保本
# 7	2018-10-08 14:58:00	买入认购 510050C1811M02500	0.1205	82	98810	194952	293762	-0.33%	2.534
# 8	2018-10-11 14:58:00	平仓认购 510050C1811M02500	0.0734	82	60188	254812	254812	-39.09%	2.439
# 9	2018-10-11 14:58:00	买入认购 510050C1811M02400	0.1229	81	99549	154939	254488	-0.32%	2.439
#10	2018-10-22 09:35:00	平仓认购 510050C1811M02400	0.1796	81	145476	300091	300091	46.14%	2.540
# No.2 买入认购后继续大跌，移仓到更低的认购，和第一条一模一样
#13	2018-10-23 14:58:00	买入认购 510050C1811M02450	0.1295	76	98420	249063	347483	-0.30%	2.513
#14	2018-10-29 14:58:00	平仓认购 510050C1811M02450	0.0890	76	67640	316399	316399	-31.27%	2.433
#15	2018-10-29 14:58:00	买入认购 510050C1811M02400	0.1144	87	99528	216523	316051	-0.35%	2.433
#16	2018-11-01 09:35:00	平仓认购 510050C1811M02400	0.1623	87	141201	357376	357376	41.87%	2.520
# 回测不移仓，保持持有的情况，不出所料的有一笔第二天止损退出了，继续修改止损比率到0.4。得到如下结论
# WinStop       = 1.4   # 止盈平仓点
# InOutLevel    = 1    # 虚实度设定。0为平值，-1为虚一档，1为实一档，以此类推
# LeftDays      = 20    # 用于选择行权日间隔 
# ==LossStop      = 0.4 区间2018.1.1-2019.5.10 最高值 561796；期末值 535693；最大回撤 -61.89%，胜率 11/13 Better
# ==LossStop      = 0.6 区间2018.1.1-2019.5.10 最高值 522260；期末值 513647；最大回撤 -44.22%，胜率 11/15 

# 2019-05-11 移仓逻辑改进，总结：比基准原代码的胜率高，总值高，原因是移仓没有移，转而固守到止盈，更加加强了“固守止盈”逻辑
# 2019-05-11 基于028模型的“固守止盈”逻辑，尝试以下2个方案，时间换空间，确保更高胜率和收益率

# No.3	2019-04-08 09:35:00	平仓认沽 510050P1904M02850	0.0343	110	37730	499646	499646	-61.89%	2.976
# 站在5月11号后视镜可以知道，这一笔其实如果继续“固守”，还能赚钱。No.1 No.2这两条都通过固守止盈退出了。
# 这涉及到期权的杠杆本质，标的随机游走，但是期权合约10倍杠杆，标的3%，期权30%当然值得做。
# 改进方案一，把到止损点的合约等金额移仓下月；有两个逻辑需要深入一下，一个是等金额，第二个是止盈点。
                # 总觉得这个还需要更多数据再看看，下次做这个逻辑。 2019-5-13
# 改进方案二，全体选择远月合约，经过20/30/10测试，远月的涨幅太小，近月时间衰减太快，20天比较合适，
#             太远月不适合作为028模型买方，除非降低止盈点，但是影响整体收益。尝试一下方案一。
# =======================

# =======================
# 实盘执行分析
# 【操作记录】5月6日 14:53 开仓LC2900@5*30:346
# 【操作记录】5月8日 09:35 平仓LC2900@5*30:170  
#            【止损出局】，太急了，LeftDays选了10，如果选下月，是5月10号反向开仓出场，-14%
# 2019-05-10 14:58:00	买入认沽 510050P1906M02850	0.1430	69	-0.28%	2.777	
# 2019-06-12 开仓条件修改为2.6%，2019年开仓机会新增2次，收益新增10万。
#            5月10号一个开仓点，实盘5月23号止盈卖出，回测一直没有到+40%，只到5月11号标的大涨，6月12号止损出局。
#            想提出的一个问题是，选择“下月实一档”合约，可以明确杠杆就是10倍，把止盈点设置到40%，对应标的4%振幅，
#            这样的逻辑似乎有问题，大涨了3%谁说就一定会跌回去？还要跌4%？
# 2019-06-12 一个模型特点摸清楚，肯定可以有效的。继续深入研究，3%的开单条件可以修改一下。虚实参数、止盈止损要综合改进。
# 2019-06-21 止损退出。郁闷得很，期权每次行情都赶不上还亏，好好分析分析模型。
# 2019-06-25 上次止损退出，20号按照信号是要买入的，25号认沽大涨，都可以止盈退出了，结果没买。强调，真正做细模型，要能重仓。
# 2020-10-12 50ETF大涨，实盘开仓
# =======================
import datetime
import jqdata
import talib 
import numpy   as np
import pandas  as pd
import matplotlib.pyplot as plt

# from pyecharts import Line
from   jqdata  import *
from   jqdata  import jy
# from   jqlib.technical_analysis import *

# =================================================
# 回测时的时间和频率设定
beginDate        = datetime.date(2021, 3, 1)
endDate          = datetime.date(2023, 8, 16)#datetime.date.today()
listHandleDates  = list(get_trade_days(start_date=beginDate,end_date=endDate))
strBuyTimebyMin  = ' 14:58:00'  # 2019-05-07 该参数经过实盘和回测验证，取值对收益率有较大影响，本质原因是尾盘波动率下降
strSellTimebyMin = ' 09:35:00'
strCheckTimebyMin2 = ' 13:35:00' # 2020-09-28 补充注释。该变量用于多次止盈止损检查。
intCommission    = 2 # 2020-08-23 手续费已调为每张2元
intInitCash      = 1000000

# 2020-09-29 增加对300ETF的回测
EtfType = '50ETF' # '300ETF'

# 追踪止损相关
HighPrice     = False # 用于买入认购的追踪止损
LowPrice      = False # 用于买入认沽的追踪止损

# 2019-06-13 关键参数
WinStop       = 1.4   # 止盈平仓点，超过成本的Win倍止盈

# 2019-06-13 关键参数。依据2018年大盘大跌，振幅较大情况下，给出60%的追踪空间，很多都扳回了，大幅涨跌时给出空间有必要。
# 2019-06-21 昨天标的大涨3.4%，方向搞反，非常郁闷，考虑到这个参数主要是适配2018年的几个过拟合，需要认真对待科学确定取值。
LossStop      = 0.6   # 追踪止损平仓点，从最高点下跌LostStop就止损

Reentry_long  = False # 追踪止损后防止重入标记
Reentry_short = False # 追踪止损后防止重入标记
CoTimer       = 0     # 计数器（用于防止止损重入）
PoTimer       = 0     # 计数器（用于防止止损重入）
TimerWindow   = 2     # 多少天接触重入限制

InOutLevel    = 1     # 虚实度设定。0为平值，-1为虚一档，1为实一档，以此类推
LeftDays      = 20    # 用于选择行权日间隔 
# 2020-10-30 将3.06修改为2.86，总盈利并没有增加，可见，放松要求增多开仓次数意义并不是太大。
OPENCONDICO   = 0.0306   # 开仓涨幅，2019-06-12之前，取值0.0306。
OPENCONDIPO   = -0.0306  # 开仓跌幅，2019-06-12之前，取值-0.0306
ret50ETF      = 0     # 用于记录开仓触发涨跌幅，注意全局变量使用范围，中途不用修改。

# =================================================
# 交易记录
colReccolumns=['date','signal', 'optcode','tradecode','price', 'num', 'amount','cash',
                    'total','ret','50ETF','openconf', '50ret','days']   
    
# 初始化，构建表结构，和初始资金，最后会删除此行    
dict={
        'date'      : beginDate,
        'signal'    : '',
        'optcode'   : '',#2
        'tradecode' : '',
        'price'     : 0.0,
        'num'       : int(0),
        'amount'    : 0.0,#6
        'cash'      : intInitCash,#12
        'total'     : intInitCash, 
        'ret'       : format(0, '.2%'),
        '50ETF'     : 0.0,
        'openconf'  : format(0, '.2%'), # 2019-06-12 新增观测参数，观测入场触发参数
        '50ret'     : format(0, '.2%'),
        'days'      : 0
     }
dfContext=pd.DataFrame(dict, columns=colReccolumns, index=[0])

# =================================================
# 功能函数 01
# 获取平值行权价。通过四舍五入取出交易所规则的行权价
# ”上证50ETF期权”的行权价格间距是3元或以下为0.05元，3元至5元（含）为0.1元，
# 5元至10元（含）为0.25元，10元至20元（含）为0.5元，
# 20元至50元（含）为1元，50元至100元（含）为2.5元，100元以上为5元。“
def getExePrice(float50ETFPrice):
    if float50ETFPrice <= 3:
        fTemp=float50ETFPrice*20
        return round(fTemp)/20
    else:
        fTemp=float50ETFPrice*10 # 3.05的计算逻辑，30.5,round一下31，31/10得到3.1
        return round(fTemp)/10
#end of getExePrice

# =================================================
# 功能函数 02
# 根据今天的日期，取出远季月
def get_next_quarter_day(dateHandle):
    # 选择的逻辑，最终是3-1，6-1，9-1，12-1这四个值
    # 先看看近季月，就是靠近当前日期最近的这几个，
    # 加入候选列表。两个要点，一是如何取年。
    # 因为近季月极可能是近月，时间价值损失太快，一律选择远季月，至少在3个月以上
    intYear  = dateHandle.year
    intMonth = dateHandle.month # date的month为int类型 10月份 10/3的值为3
    
    if (intMonth-1)/3 == 3: # 10月，11月，12月第四季度
        intYear  = intYear + 1
        intMonth = 3   # 下一年的3月
        
    else:              # 其他月份，比如1/2/3月份，(M-1)/3==0，取6月，
                       # 4/5/6月份，(M-1)/3==1，取9月
                       # 7/8/9月份，(M-1)/3==2，取12月
        intMonth = 3 * ((int)((intMonth-1)/3)+2) 
        
    return datetime.date(intYear,intMonth,1)

# =================================================
# 功能函数 03
# 根据平值行权价，指定的虚实度Level参数，获取模型逻辑所需的虚实度行权价
# 2019-05-21 发现计算3.2的虚一档时，逻辑有错误，需要修正。
# 由于跨档间隔不同，逻辑还是比较复杂，注释详细如下
def setInOutLevel(strType, fExePrice, InOutLevel):
    if strType == 'CO':
        # 换仓前后都在同样的间隔，则直接使用间隔
        if fExePrice >= 3:
            fExePrice -= 0.1*InOutLevel
            # 不同间隔，则需要缩放该区间的间隔
            if fExePrice < 3:
                fExePrice = 3 - (3-fExePrice)/2
        else:
        # <3的区间
            fExePrice -= 0.05*InOutLevel
            if fExePrice >= 3:
                fExePrice = 3 + (fExePrice-3)*2            
    elif strType == 'PO':
        if fExePrice >= 3:
            fExePrice += 0.1*InOutLevel
            if fExePrice < 3:
                fExePrice = 3 - (3-fExePrice)/2            
        else:
            fExePrice += 0.05*InOutLevel
            if fExePrice >= 3:
                fExePrice = 3 + (fExePrice-3)*2             
        
    # 2019-05-21 发现计算3.2的虚一档时，逻辑有错误，需要修正。
    # if fExePrice > 3:
        # fExePrice = 3 + (fExePrice-3)*2
    return fExePrice
# end of def setInOutLevel(strType, fExePrice, InOutLevel):
   
# =================================================
# 功能函数04
# 2020-09-28 移植自option031【定期03】定周期卖出认沽
# 获取指定时间内50ETF行权日期
def get_exercise_days(start_date, end_date):
    # 2020-09-28 添加关于OPT_CONTRACT_INFO注释。https://www.joinquant.com/help/api/help?name=Option
    # 聚宽关于期权信息query，有两个常用的表，OPT_CONTRACT_INFO和OPT_DAILY_PREOPEN
    # 其一、query(opt.OPT_CONTRACT_INFO)：表示从opt.OPT_CONTRACT_INFO这张表中查询期权基本资料数据，可以查询所以存在过的合约信息
    # 其二、OPT_DAILY_PREOPEN：表示从opt.OPT_DAILY_PREOPEN这张表中查询期权每日盘前静态数据，可以查询在特定日期存续的合约
    # 本函数需要判定指定回测时间内的所有行权日日期，必须调用OPT_CONTRACT_INFO
    q=query(
            opt.OPT_CONTRACT_INFO.exercise_date     #exercise_date	str	行权日	2019-02-27 
            ).filter(opt.OPT_CONTRACT_INFO.underlying_symbol=='510050.XSHG',
                     opt.OPT_CONTRACT_INFO.contract_type=='CO',
                     opt.OPT_CONTRACT_INFO.exercise_date>=start_date,
                     opt.OPT_CONTRACT_INFO.exercise_date<=end_date
                     ).order_by(opt.OPT_CONTRACT_INFO.exercise_date.asc())
                     
    # 经过更多的filter条件，选出较少较精确的条目，注意聚宽有3000条限制
    dfConInfo=opt.run_query(q)
    
    if len(dfConInfo)==0:
        print("==get_exercise_days(): 查询参数错误==")
        return list()
    
    return list(dfConInfo['exercise_date'].drop_duplicates())
# end of get_exercise_days

listExeDates = get_exercise_days(start_date=beginDate,end_date=endDate)
   
# =================================================
# 功能函数 06
# 追踪止盈止损
# 参数：strTrailTime。strTrailTime1  = tradeDate.strftime("%Y-%m-%d")+ gTrailTime
#       由日期和时分秒组成，调用前注意赋值。
def TrailingCheck(strTrailTime):
    global dfContext
    global HighPrice
    global LowPrice
    global Reentry_short
    global Reentry_long
    
    # 开仓后HighPrice置为有效，平仓后置为False
    if HighPrice:
        print('/*--LC头寸的Trailing_Stop开启中--*/'), strTrailTime
        strCoCode     = dfContext.iloc[-1]['optcode'] # LC头寸合约代码
        fltCostPrice  = dfContext.iloc[-1]['price']   # LC头寸成本
        
        # strTrailTime  = tradeDate.strftime("%Y-%m-%d")+ strSellTimebyMin
        # 2020-09-29 添加注释。追踪检查时LC头寸的价格
        fltCurCoPrice = get_price(strCoCode, strTrailTime,strTrailTime,'1m', ['close'], fq=None).iloc[0,0]    
        HighPrice     = max(HighPrice, fltCurCoPrice) # 刷新LC头寸最高价，用于追踪止损
        
        # 2019-06-13 计算标的幅度，如果标的幅度达标，止盈
        flt50ETFPrice = dfContext.iloc[-1]['50ETF'] # 用于判断检查时点标的的涨跌情况
        flt50ETFCur   = get_price('510050.XSHG', strTrailTime,strTrailTime,'1m', ['close'], fq=None).iloc[0,0]        
        
        print("fltCurCoPrice : "), fltCurCoPrice
        print("fltCostPrice  : "), fltCostPrice
        print("【Cur/Cost-1】: "), format(fltCurCoPrice/fltCostPrice-1, '.2%')
        print("HighPrice     : "), HighPrice
        
        # 2019-06-13 修改止盈判定参数
        # 2020-08-22 2020年2月3号开年第一天开盘，疫情肆虐，大盘暴跌“偏误”，开仓LC，这种情况下按照2019-06-13的止盈条件（标的涨3%）
        # 就止盈，显然太早了，发生“偏误”的程度不一，每次止盈的参数硬性一模一样很显然不合适。
        # 2020-08-23 修正模型，追求更高的盈亏比，放飞上限，而不是单纯追求胜率，一次十倍抵得上多次的胜率。
        if fltCurCoPrice  > fltCostPrice*WinStop:
        # if fltCurCoPrice  > fltCostPrice*WinStop or flt50ETFCur > flt50ETFPrice*1.03:
            print('/*==止盈平仓LC头寸==*/')
            pingcang_LC(tradeDate, strTrailTime)
        elif fltCurCoPrice < HighPrice*(1-LossStop):
            print('/*==Training止损平仓LC==*/')
            pingcang_LC(tradeDate, strTrailTime)
            # 追踪止损后，必须防止重入
            # Reentry_long = True
        print('/*-------------------------------*/')
    if LowPrice:
        print('/*--LP头寸的Trailing_Stop开启中--*/'), strTrailTime
        strPoCode     = dfContext.iloc[-1]['optcode']
        fltCostPrice  = dfContext.iloc[-1]['price']
        
        # strTrailTime  = tradeDate.strftime("%Y-%m-%d")+ strSellTimebyMin
        fltCurPoPrice = get_price(strPoCode, strTrailTime,strTrailTime,'1m', ['close'], fq=None).iloc[0,0]           
        LowPrice      = max(LowPrice, fltCurPoPrice) # 还是保持合约的最高价格
        
        # 2019-06-13 计算标的幅度，如果标的幅度达标，止盈
        flt50ETFPrice = dfContext.iloc[-1]['50ETF']
        flt50ETFCur   = get_price('510050.XSHG', strTrailTime,strTrailTime,'1m', ['close'], fq=None).iloc[0,0]         
        
        print("fltCurPoPrice : "), fltCurPoPrice
        print("fltCostPrice  : "), fltCostPrice
        print("【Cur/Cost-1】: "), format(fltCurPoPrice/fltCostPrice-1, '.2%')
        print("LowPrice      : "), LowPrice
        
        # 2019-06-13 观察到实盘5月10号开仓，标的方向已经大于3%，但是由于波动率下降，达不到WinStop导致最后止损
        # 2019-06-13 必须改进此处逻辑，028模型是delta模型，标的涨幅已达到统计止盈范畴，需要退出。
        # if fltCurPoPrice  > fltCostPrice*WinStop:
        # 2020-09-28 flt50ETFCur < flt50ETFPrice*0.97这个条件的止盈，在2020-07-16早上触发，却浪费了下午大跌的利润，
        # 如果用WinStop在这里显然收益更大，但是有些情况是波动率下降，标的幅度达到但是合约价格没涨太多的情况，导致没有止盈的回撤
        # 这两种止盈情况可以更细化一些，提高整体收益。
        if fltCurPoPrice  > fltCostPrice*WinStop:
            print('/*==触发收益率WinStop，止盈平仓LP==*/')
            pingcang_LP(tradeDate, strTrailTime)
        elif  flt50ETFCur < flt50ETFPrice*0.97:  # 2020-09-28 这里既然是跌幅回归，采用0.97的固定值是不合理的，是否可以采用开仓涨幅？
            print('/*==触发标的跌幅回归，止盈平仓LP==*/')
            pingcang_LP(tradeDate, strTrailTime)
        elif fltCurPoPrice < LowPrice*(1-LossStop):
            print('==Trailing止损平仓买入认沽==')
            pingcang_LP(tradeDate, strTrailTime)
            # Reentry_short = True
        print('/*-------------------------------*/')
# end of TrailingCheck(strTrailTime)

# =================================================
# 交易函数01 认购开仓
# =================================================
def kaicang_LC(dateHandle):
    global dfContext
    global colReccolumns
    global intCommission
    global strBuyTimebyMin
    global HighPrice
    global Timer
    
    minuteBuy       = dateHandle.strftime("%Y-%m-%d")+strBuyTimebyMin
    dp50ETFPrice    = get_price('510050.XSHG', minuteBuy, minuteBuy, '1m', ['close'], fq=None)
    float50ETFPrice = dp50ETFPrice.iloc[0]['close']
    
    # 平值行权价
    fExePrice = getExePrice(float50ETFPrice)
    # 根据全局参数InOutLevel，选取目标行权价
    fConExe   = setInOutLevel('CO', fExePrice, InOutLevel)
    print("平值行权价："), fExePrice    
    print("指定虚实度行权价："), fConExe  

    # 确定下季月，时间进行过滤，找到目标合约号
    # dateNextQuarterDay = get_next_quarter_day(dateHandle)

    # 2020-11-09 根据指定虚实度和指定行权时间间隔获取符合条件的期权合约
    # OPT_DAILY_PREOPEN的获取方法，会过滤掉A修正类合约
    q=query(
            opt.OPT_DAILY_PREOPEN.code,             #code	        str	合约代码	10001313.XSHG；
            opt.OPT_DAILY_PREOPEN.trading_code,     #trading_code	int	合约交易代码	510050C1810M02800
            ).filter(opt.OPT_DAILY_PREOPEN.underlying_symbol=='510050.XSHG', 
                     opt.OPT_DAILY_PREOPEN.exercise_price==fConExe,
                     opt.OPT_DAILY_PREOPEN.date==dateHandle,
                     opt.OPT_DAILY_PREOPEN.contract_type=='CO',
                     opt.OPT_DAILY_PREOPEN.exercise_date>dateHandle+datetime.timedelta(days=LeftDays),
                     opt.OPT_DAILY_PREOPEN.exercise_date<dateHandle+datetime.timedelta(days=160)
                     ).order_by(opt.OPT_DAILY_PREOPEN.exercise_date.asc())
    dfConInfo=opt.run_query(q)
    if len(dfConInfo)==0:
        print("==行权合约选择逻辑发生错误==")
        # 以下逻辑2019-05-10添加，原因是大涨大跌，导致新合约没挂出来。
        print("==调整到平值重新query==")    
        q=query(
                opt.OPT_DAILY_PREOPEN.code,             #code	        str	合约代码	10001313.XSHG；
                opt.OPT_DAILY_PREOPEN.trading_code,     #trading_code	int	合约交易代码	510050C1810M02800
                ).filter(opt.OPT_DAILY_PREOPEN.underlying_symbol=='510050.XSHG', 
                         opt.OPT_DAILY_PREOPEN.date==dateHandle, 
                         opt.OPT_DAILY_PREOPEN.exercise_price==fExePrice,
                         opt.OPT_DAILY_PREOPEN.contract_type=='CO',
                         opt.OPT_DAILY_PREOPEN.exercise_date>dateHandle+datetime.timedelta(days=LeftDays),
                         opt.OPT_DAILY_PREOPEN.exercise_date<dateHandle+datetime.timedelta(days=160)                 
                         ).order_by(opt.OPT_DAILY_PREOPEN.exercise_date.asc())
        dfConInfo=opt.run_query(q)           
    
    strCoCode       = dfConInfo.iloc[0]['code']
    strCoTradeCode  = dfConInfo.iloc[0]['trading_code']

    # 2，价格，获取一个期权的价格，strBuyTimebyMin可指定
    strBuyTime      = dateHandle.strftime("%Y-%m-%d")+ strBuyTimebyMin
    dpPPricebyMin   = get_price(strCoCode, strBuyTime, strBuyTime, '1m', ['close'], fq=None)
    fltCoPrice      = dpPPricebyMin.iloc[0]['close']
    if math.isnan(fltCoPrice): # 聚宽返回的nan必须使用这个函数判断
        print("===合约价格获取失败===")
        print("strCoCode      :", strCoCode)
        print("strCoTradeCode :", strCoTradeCode)        
        return
    fltCash     = dfContext.iloc[-1]['cash']
    iNumCo      = int(intInitCash/(fltCoPrice*10000+intCommission))
    fltCoAmount = iNumCo*fltCoPrice*10000
    fltCash     = int(fltCash-fltCoAmount-intCommission*(iNumCo)) 
    fltTotal    = fltCash+fltCoAmount
    fltRet      = -float(intCommission*iNumCo)/intInitCash
    print('fltRet: '),fltRet
    # 写入记录dfContext，同一天的买卖记录到一条记录

    # 写入记录
    dict={
            'date'      : minuteBuy,
            'signal'    : '买入认购',
            'optcode'   : strCoCode,
            'tradecode' : strCoTradeCode,
            'price'     : fltCoPrice,
            'num'       : iNumCo,
            'amount'    : fltCoAmount,#6
            'cash'      : fltCash,#12
            'total'     : fltTotal, 
            'ret'       : format(fltRet, '.2%'),
            '50ETF'     : float50ETFPrice,
            'openconf'  : format(ret50ETF, '.2%'), # 2019-06-12 新增观测参数，观测入场触发参数
            '50ret'     : '', #format(0, '.2%'),
            'days'      : 0            
         }                            
    dfNewRec  = pd.DataFrame(dict, columns=colReccolumns, index=[len(dfContext)])
    print(dfNewRec)
    # pieces    = [dfContext, dfNewRec]
    # dfContext = pd.concat(pieces)
    dfContext = dfContext.append(dfNewRec)
    HighPrice = fltCoPrice # 用于CoCode追踪止损
    CoTimer     = 0 # 把以前计算器清零，开始追踪则重新开始计数
# end of kaicang_LC()

# =================================================
# 交易函数02
# =================================================
def pingcang_LC(dateSell, strpingcangTime): 
    global dfContext
    global colReccolumns
    global intCommission   
    global strSellTimebyMin
    global HighPrice    

    fltCash        = dfContext.iloc[-1]['cash']
    strCoCode      = dfContext.iloc[-1]['optcode']
    strCoTradeCode = dfContext.iloc[-1]['tradecode']   
    iNumCo         = dfContext.iloc[-1]['num']

    # 获取平仓时刻品种价格
    # strSellTime     = dateSell.strftime("%Y-%m-%d")+strpingcangTime
    dpPPricebyMin   = get_price(strCoCode,strpingcangTime,strpingcangTime,'1m',['close'], fq=None)
    fltCoPrice      = dpPPricebyMin.iloc[0]['close']
    fltCoAmount     = fltCoPrice*iNumCo*10000
    fltCash         = fltCash+fltCoAmount-intCommission*iNumCo
    fltTotal        = fltCash
    fltRet          = (fltCoPrice-intCommission/10000)/(dfContext.iloc[-1]['price'])-1
    # 取交易时刻50ETF的价格
    dp50ETFPrice    = get_price('510050.XSHG', strpingcangTime,strpingcangTime,'1m', ['close'], fq=None)
    float50ETFPrice = dp50ETFPrice.iloc[0]['close']
    
    # 2019-06-12 观测头寸持有天数
    dateEnd   = datetime.datetime.strptime(strpingcangTime, '%Y-%m-%d %H:%M:%S')
    dateBegin = datetime.datetime.strptime(dfContext.iloc[-1]['date'], '%Y-%m-%d %H:%M:%S')
    
    dict={
            'date'      : strpingcangTime,
            'signal'    : '平仓认购',
            'optcode'   : strCoCode,
            'tradecode' : strCoTradeCode,
            'price'     : fltCoPrice,
            'num'       : iNumCo,
            'amount'    : fltCoAmount,#6
            'cash'      : fltCash,#12
            'total'     : fltTotal, 
            'ret'       : format(fltRet, '.2%'),
            '50ETF'     : float50ETFPrice,
            'openconf'  : '',#format(0, '.2%'), # 2019-06-12 新增观测参数，观测入场触发参数
            '50ret'     : format(float50ETFPrice/dfContext.iloc[-1]['50ETF']-1, '.2%'), # 2019-06-12 新增观测参数，观测周期50ETF涨跌幅
            'days'      : (dateEnd-dateBegin).days   # 2019-06-12 新增观测参数，观测入场触发参数
         } 
    dfNewRec  = pd.DataFrame(dict, columns=colReccolumns, index=[len(dfContext)])
    print(dfNewRec)
    # pieces    = [dfContext, dfNewRec]
    # dfContext = pd.concat(pieces)
    dfContext = dfContext.append(dfNewRec)
    HighPrice = False # 平仓认购，追踪置为False
# end of pingcang_LC()    

# =================================================
# 交易函数03
# 认沽开仓
# =================================================
def kaicang_LP(dateHandle):
    global dfContext
    global colReccolumns
    global intCommission
    global strBuyTimebyMin
    global LowPrice
    global Timer

    minuteBuy       = dateHandle.strftime("%Y-%m-%d")+strBuyTimebyMin
    dp50ETFPrice    = get_price('510050.XSHG', minuteBuy, minuteBuy, '1m', ['close'], fq=None)
    float50ETFPrice = dp50ETFPrice.iloc[0]['close']
    fExePrice       = getExePrice(float50ETFPrice)
    fConExe         = setInOutLevel('PO', fExePrice, InOutLevel)
    print("平值行权价："), fExePrice    
    print("指定虚实度行权价："), fConExe      
    
    # 根据平值行权价fExePrice进行过滤，时间，时间可以比较过滤，非常好，20倍提升过滤效率
    q=query(
            opt.OPT_DAILY_PREOPEN.code,             #code	        str	合约代码	10001313.XSHG；
            opt.OPT_DAILY_PREOPEN.trading_code,     #trading_code	int	合约交易代码	510050C1810M02800
            ).filter(opt.OPT_DAILY_PREOPEN.underlying_symbol=='510050.XSHG', 
                     opt.OPT_DAILY_PREOPEN.date==dateHandle, 
                     opt.OPT_DAILY_PREOPEN.exercise_price==fConExe,
                     opt.OPT_DAILY_PREOPEN.contract_type=='PO',
                     opt.OPT_DAILY_PREOPEN.exercise_date>dateHandle+datetime.timedelta(days=LeftDays),
                     opt.OPT_DAILY_PREOPEN.exercise_date<dateHandle+datetime.timedelta(days=160)                 
                     ).order_by(opt.OPT_DAILY_PREOPEN.exercise_date.asc())
    dfConInfo=opt.run_query(q)
    if len(dfConInfo)==0:
        print("==行权合约选择逻辑发生错误==")
        # 以下逻辑2019-05-10添加，原因是2019-02-25,50大涨7%，导致新合约没挂出来。
        print("==调整到平值重新query==")    
        q=query(
                opt.OPT_DAILY_PREOPEN.code,             #code	        str	合约代码	10001313.XSHG；
                opt.OPT_DAILY_PREOPEN.trading_code,     #trading_code	int	合约交易代码	510050C1810M02800
                ).filter(opt.OPT_DAILY_PREOPEN.underlying_symbol=='510050.XSHG', 
                         opt.OPT_DAILY_PREOPEN.date==dateHandle, 
                         opt.OPT_DAILY_PREOPEN.exercise_price==fExePrice,
                         opt.OPT_DAILY_PREOPEN.contract_type=='PO',
                         opt.OPT_DAILY_PREOPEN.exercise_date>dateHandle+datetime.timedelta(days=LeftDays),
                         # 2月份是不挂4月份合约的，5月份不挂7月份合约
                         opt.OPT_DAILY_PREOPEN.exercise_date<dateHandle+datetime.timedelta(days=160)
                         ).order_by(opt.OPT_DAILY_PREOPEN.exercise_date.asc())
        dfConInfo=opt.run_query(q)
        print("==调整到平值重新query到的合约df==")          
        print(dfConInfo)
        print(LeftDays)
        print(fExePrice)
        # print(dateHandle+datetime.timedelta(days=20))
        # print(dateHandle+datetime.timedelta(days=60))
        # return
    
    strPoCode       = dfConInfo.iloc[0]['code']
    strPoTradeCode  = dfConInfo.iloc[0]['trading_code']
    strBuyTime      = dateHandle.strftime("%Y-%m-%d")+ strBuyTimebyMin
    dpPPricebyMin   = get_price(strPoCode,strBuyTime,strBuyTime,'1m',['close'], fq=None)
    fltPoPrice      = dpPPricebyMin.iloc[0]['close']
    if math.isnan(fltPoPrice): # 聚宽返回的nan必须使用这个函数判断
        print("===合约价格获取失败===")
        print("strPoCode      :", strPoCode)
        print("strPoTradeCode :", strPoTradeCode)
        return    
    fltCash         = dfContext.iloc[-1]['cash']
    iNumPo          = int(intInitCash/(fltPoPrice*10000+intCommission))
    fltPoAmount     = iNumPo*fltPoPrice*10000
    fltCash         = fltCash-fltPoAmount-intCommission*(iNumPo) # 手续费每张4块钱
    fltTotal        = fltCash+fltPoAmount
    fltRet          = -float(intCommission*iNumPo)/intInitCash
    dict={
            'date'      : minuteBuy,
            'signal'    : '买入认沽',
            'optcode'   : strPoCode,
            'tradecode' : strPoTradeCode,
            'price'     : fltPoPrice,
            'num'       : iNumPo,
            'amount'    : fltPoAmount,#6
            'cash'      : fltCash,#12
            'total'     : fltTotal, 
            'ret'       : format(fltRet, '.2%'),
            '50ETF'     : float50ETFPrice,
            'openconf'  : format(ret50ETF, '.2%'), # 2019-06-12 新增观测参数，观测入场触发参数
            '50ret'     : '',#format(0, '.2%'),
            'days'      : 0                   
         }                            
    dfNewRec  = pd.DataFrame(dict, columns=colReccolumns, index=[len(dfContext)])
    print(dfNewRec)    
    # pieces    = [dfContext, dfNewRec]
    # dfContext = pd.concat(pieces)
    dfContext = dfContext.append(dfNewRec)
    LowPrice  = fltPoPrice
    PoTimer     = 0 # 把以前计算器清零，开始追踪则重新开始计数
# end of kaicang_LP()

# =================================================
# 交易函数04
# =================================================
def pingcang_LP(dateSell, strpingcangTime): 
    global dfContext
    global colReccolumns
    global intCommission   
    global strSellTimebyMin
    global LowPrice

    strPoCode       = dfContext.iloc[-1]['optcode']
    strPoTradeCode  = dfContext.iloc[-1]['tradecode']   
    iNumPo          = dfContext.iloc[-1]['num']
    # strSellTime     = dateSell.strftime("%Y-%m-%d")+strpingcangTime
    dpPPricebyMin   = get_price(strPoCode,strpingcangTime,strpingcangTime,'1m',['close'], fq=None)
    fltPoPrice      = dpPPricebyMin.iloc[0,0]#]['close']
    if math.isnan(fltPoPrice): # 聚宽返回的nan必须使用这个函数判断
        print("===合约价格获取失败===")
        return        
    fltPoAmount     = fltPoPrice*iNumPo*10000
    fltCash         = dfContext.iloc[-1]['cash']
    fltCash         = fltCash+fltPoAmount-intCommission*iNumPo
    fltTotal        = fltCash
    fltRet          = (fltPoPrice-intCommission/10000)/(dfContext.iloc[-1]['price'])-1

    dp50ETFPrice    = get_price('510050.XSHG', strpingcangTime,strpingcangTime,'1m', ['close'], fq=None)
    float50ETFPrice = dp50ETFPrice.iloc[0,0]
    
    # 2019-06-12 观测头寸持有天数
    dateEnd   = datetime.datetime.strptime(strpingcangTime, '%Y-%m-%d %H:%M:%S')
    dateBegin = datetime.datetime.strptime(dfContext.iloc[-1]['date'], '%Y-%m-%d %H:%M:%S')    
    
    dict={
            'date'      : strpingcangTime,
            'signal'    : '平仓认沽',
            'optcode'   : strPoCode,
            'tradecode' : strPoTradeCode,
            'price'     : fltPoPrice,
            'num'       : iNumPo,
            'amount'    : fltPoAmount,#6
            'cash'      : fltCash,#12
            'total'     : fltTotal, 
            'ret'       : format(fltRet, '.2%'),
            '50ETF'     : float50ETFPrice,
            'openconf'  : '',#format(0, '.2%'), # 2019-06-12 新增观测参数，观测入场触发参数
            '50ret'     : format(float50ETFPrice/dfContext.iloc[-1]['50ETF']-1, '.2%'), # 2019-06-12 新增观测参数，观测周期50ETF涨跌幅
            'days'      : (dateEnd-dateBegin).days   # 2019-06-12 新增观测参数，观测入场触发参数            
         } 
    dfNewRec    = pd.DataFrame(dict, columns=colReccolumns, index=[len(dfContext)])
    print(dfNewRec)
    # pieces      = [dfContext, dfNewRec]
    # dfContext   = pd.concat(pieces) 
    dfContext = dfContext.append(dfNewRec)
    LowPrice    = False
# end of pingcang_LP() 



# 交易日遍历主循环函数
for i in range(1, len(listHandleDates)):
    tradeDate = listHandleDates[i]
    Cross     = 0
    Signal    = 0
    print("/*==============================================================================*/")
    print("                    交易日期     : "), tradeDate
    
    # =============== 追踪止损模块，每个交易日都必须进行判断 =============== #
    strTrailTime  = tradeDate.strftime("%Y-%m-%d")+ strSellTimebyMin
    TrailingCheck(strTrailTime)
    # 2020-09-29 以下代码用于多次止损检查，经过测试发现效果不佳，越多的注释波动肯定是越大的，过于频繁触发止损点
    # strTrailTime2  = tradeDate.strftime("%Y-%m-%d")+ strCheckTimebyMin2
    # TrailingCheck(strTrailTime2)    
    # =============== 追踪止损模块 END =============== #   

    # 2020-09-28 增加一直没有被止盈止损触发，在行权日即将到期必须平仓合约的逻辑
    if (tradeDate in listExeDates):
        # 在行权日尚有LP头寸
        if dfContext.iloc[-1]['signal'] == '买入认沽':
            # 以下条件避免靠近行权日时误平仓下月合约。利用合约存续天数来判断。
            dateHandleDay     = datetime.datetime.strptime(strTrailTime,               '%Y-%m-%d %H:%M:%S')
            dateContractBegin = datetime.datetime.strptime(dfContext.iloc[-1]['date'], '%Y-%m-%d %H:%M:%S')
            # print("20200928调试行权日平仓，头寸存续天数:", (dateHandleDay - dateContractBegin).days)
            if (dateHandleDay - dateContractBegin).days > LeftDays:
                # 平仓到期的LP头寸
                print("==平仓到期的LP头寸==")
                pingcang_LP(tradeDate, strTrailTime)
        elif dfContext.iloc[-1]['signal'] == '买入认购':
            dateHandleDay     = datetime.datetime.strptime(strTrailTime,               '%Y-%m-%d %H:%M:%S')
            dateContractBegin = datetime.datetime.strptime(dfContext.iloc[-1]['date'], '%Y-%m-%d %H:%M:%S')
            if (dateHandleDay - dateContractBegin).days > LeftDays:
                # 平仓到期的LC头寸
                print("==平仓到期的LC头寸==")
                pingcang_LC(tradeDate, strTrailTime)
            
    # 判断入场逻辑
    print('/*-------------涨幅偏误入场逻辑判断-------------*/')
    df50ETF     = get_price('510050.XSHG', end_date=tradeDate, count=2, fields=['open', 'high', 'low', 'close'], fq=None)
    strBuyTime  = tradeDate.strftime("%Y-%m-%d")+ strBuyTimebyMin
    fltCurPrice = get_price('510050.XSHG', strBuyTime,strBuyTime,'1m', ['close'], fq=None).iloc[0,0] 
    # 2020-10-30 根据老赵提议，试试尾盘和开盘的比值。结论是开仓次数下降很多，而且胜率也下降不少，主要还是因为偏离不够
    # 028模型就是做情绪偏离，还是坚守之前的逻辑。
    ret50ETF    = fltCurPrice/(df50ETF.iloc[-2]['close'])-1 # 备份
    # ret50ETF    = fltCurPrice/(df50ETF.iloc[-1]['open'])-1
    print("当前50ETF: "), fltCurPrice
    print("昨收50ETF: "), df50ETF.iloc[-2]['close']
    # print("今开50ETF: "), df50ETF.iloc[-1]['open']
    print("涨幅     : "), format(ret50ETF, '.2%')    
    
    # 当日涨跌幅度超过3%作为开仓条件
    if ret50ETF>OPENCONDICO:
        Cross = 1
    elif ret50ETF<OPENCONDIPO:
        Cross = -1
    else:
        Cross = 0 
        
    #判断交易信号：均线交叉+可二次入场条件成立
    if  Cross == 1: 
        if Reentry_long == False:
            Signal = 1
        else:
            print('Reentry_long Limited')
    elif Cross == -1:
        if Reentry_short == False:
            Signal = -1
        else:
            print('Reentry_short Limited')            
    else:
        Signal = 0
        
    # print("Cross      : "), Cross
    # print("Signal     : "), Signal  
    print('/*-------------涨幅偏误入场逻辑判断-------------*/')    

    if Signal == -1: # 标的大跌，准备反向，开仓LC
        # 181018和181019冰火两重天，必须在买入前确保卖出
        if dfContext.iloc[-1]['signal'] == '买入认沽':
            # 收盘前平仓使用收盘前的时间
            print("==反手平仓LP==")
            pingcang_LP(tradeDate, strBuyTime)
        elif dfContext.iloc[-1]['signal'] == '买入认购':
            # 收盘前平仓使用收盘前的时间
            print("==平仓，移仓==")
            # 2019-05-11 回测不移仓情况下，胜率和收益率比较。注释以下平仓逻辑。
            # pingcang_LC(tradeDate, strBuyTime)            
        if dfContext.iloc[-1]['signal']!='买入认购':
            print("/*==标的大跌，开仓LC==*/")
            kaicang_LC(tradeDate)
    elif Signal == 1: # 标的大涨，准备反向，开仓LP
        if dfContext.iloc[-1]['signal'] == '买入认购':
            print("==反手平仓LC==")
            pingcang_LC(tradeDate, strBuyTime)
        if dfContext.iloc[-1]['signal']!='买入认沽':
            print("==买入认沽看跌==")
            kaicang_LP(tradeDate)
# enf of 主循环函数

# 操作完成，删除第一行的初始化数据
dfContext=dfContext.drop([0])
dfContext.to_csv('028.csv', mode='w', index=True, encoding='utf_8_sig')

# =================================================
# 功能函数
# anynisis_positions 分析开仓后头寸的走势
# 2020-10-16 如何在作图的时候去掉收盘和周末的无效价格，这里有看出matplotlib的局限，不好用
# 2020-10-16 研究pyecharts的使用，在pycharm中试验成功，可惜“jq研究”不支持pyecharts。
# 2020-10-16 开通jqdatasdk可以在本地电脑进行研究，准备结合echart研究，可惜opt相关环境和线上不一致产生问题
# =================================================
def anynisis_positions(dfPositions):
    strStartTime = dfPositions['date'][1]     #'2019-3-23'
    strEndTime   = dfPositions['date'][2] #'2019-5-26'
    strTradingCode = dfPositions['tradecode'][1]
    strStockCode = dfPositions['optcode'][1]

    print(strStartTime)
    dfObserver = get_price(strStockCode, strStartTime, strEndTime, '30m',['open','close','low','high'])
    # dfObserver = get_price(strStockCode, strStartTime, strEndTime, '30m', ['close'])
    # dfETF      = get_price('510050.XSHG', strStartTime, strEndTime, '30m', ['close'])
    
    # dfObserver['50ETF'] = dfETF['close']
    print(dfObserver)
    dfObserver.to_csv('dfPosiitions.csv')

    # fig = plt.figure(figsize=(20,8))
    # ax = fig.add_subplot(111)
    
    # 2020-10-16 横坐标优化，matplotlib没有现成的跳过空白时间的函数，提出的解决方案也不好，基础功能要重写是不科学的
    # 尝试一下其他的画图工具吧pyecharts

    # line = Line("开仓头寸观察")
    # is_label_show是设置上方数据是否显示
    # line.add(strTradingCode, dfObserver.index, dfObserver['close'], is_label_show=True)
    # line.add(u'50ETF', dfObserver.index, dfObserver['50ETF'], is_label_show=True)
    # line.render()    


    # 加载数据
    # ax.grid()
    # lns1 = ax.plot(dfObserver.index, dfObserver['close'], '-r', label = strTradingCode)

    # ax2 = ax.twinx()
    # lns2 = ax2.plot(dfObserver.index, dfObserver['50ETF'], '-b', label = u'50ETF')

    # lns = lns1+lns2
    # labs = [l.get_label() for l in lns]
    # ax.legend(lns, labs, loc=2)

    # ax.set_xlabel(u"tradedate")
    # ax.set_ylabel(r"close")
    # plt.title(strTradingCode)
    # plt.show()    
    return

#end of def anynisis_positions():
# 2020-10-16 添加头寸详细信息观察功能
# anynisis_positions(dfContext)

# option028【择时06】“过度偏误”策略
# end of the file