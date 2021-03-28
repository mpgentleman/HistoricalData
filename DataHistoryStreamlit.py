######################################
#
#   Import Modules
#
######################################
import quandl
import pandas as pd
import plotly.express as px
quandl.ApiConfig.api_key = 'pkEYWsQ3hyKiEDZbUV8X'
import plotly as plotly
import plotly.figure_factory as ff
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import cufflinks as cf
cf.go_offline()
import datetime
import seaborn as sns
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np
import math
import streamlit as st
from matplotlib import colors

#########################################
#
#  Define Functions
#
#########################################

def background_gradient(s, m, M, cmap='PuBu', low=0, high=0):


    rng = M - m
    norm = colors.Normalize(m - (rng * low),
                            M + (rng * high))
    normed = norm(s.values)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ['background-color: %s' % color for color in c]

def get_Bin_Distribution(DF,price_buckets):


    
    cut_labels_4=['ExtremeTail-','ModerateTail-', 'CrazyShit-', 'DayMove-','DayMove+', 'CrazyShit+', 'ModerateTail+','ExtremeTail+']
    cut_labels_Fut=['ExtremeTail-','ModerateTail-', 'CrazyShit-', 'DayMove-','DayMove+', 'CrazyShit+', 'ModerateTail+','ExtremeTail+']
    cut_labels_4R=cut_labels_4[::-1]
    DF['Daily_Vol_Bins']=pd.cut(DF['Daily_Vol_Move'], bins=np.linspace(-20, 20, 9), labels=cut_labels_4)
    DF['Daily_Futures_Bins'] = pd.cut(DF['Daily_Fut_Move'],price_buckets , labels=cut_labels_Fut)

    list4=[]
    list_outerR=[]
    for i in range(len(cut_labels_4R)):
    
        for t in range(len(cut_labels_Fut)):
            list4.append(DF[DF['Daily_Vol_Bins'].isin(cut_labels_4R[i:i+1])&DF['Daily_Futures_Bins'].isin(cut_labels_Fut[t:t+1])]['AtM'].count())
        list_outerR.append(list4)
        list4=[]

    heat_df=pd.DataFrame(list_outerR,columns=cut_labels_Fut,index=cut_labels_4R)

    #display(heat_df)

    cm = sns.diverging_palette(1, 250, as_cmap=True)
    even_range = np.max([np.abs(heat_df.values.min()), np.abs(heat_df.values.max())])
    heat_df.style.apply(background_gradient,
               cmap=cm,
               m=-even_range,
               M=even_range).set_precision(2)
    return(heat_df)
               
def get_Line_Chart(DF,Other_Chart_Values_options):


    for i in range(len(Other_Chart_Values_options)):
        titleIs=column_to_Chart[i]+' History'
        fig3=DF[column_to_Chart[i]].iplot(title=titleIs,asFigure=True)
        st.plotly_chart(fig3)
def get_Cloud_Chart(DF,Other_Chart_Values_options):


    for i in range(len(Other_Chart_Values_options)):
        titleIs=column_to_Chart[i]+' History'
        #fig3=DF[column_to_Chart[i]].iplot(title=titleIs,asFigure=True)
        fig3=DF.iplot(kind="scatter",x="Future",y=column_to_Chart[i], mode ='markers',title=titleIs,asFigure=True)
        st.plotly_chart(fig3)
def get_Path_Chart(DF,Other_Chart_Values_options):


    for i in range(len(Other_Chart_Values_options)):
        titleIs=column_to_Chart[i]+' History'
        #fig3=DF[column_to_Chart[i]].iplot(title=titleIs,asFigure=True)
        fig3=DF.iplot(kind="scatter",x="Future",y=column_to_Chart[i] ,title=titleIs,asFigure=True)
        st.plotly_chart(fig3)

def transform_Text_Futures_File(DF):


    DF=DF[['Date',' Open',' High',' Low',' Last']]

    DF.columns=['Date','Open','High','Low','Close']
    DF['Date'] = pd.to_datetime(DF['Date'])
    DF.set_index("Date", inplace=True)
    DF['Daily_Fut_Move']=DF['Close'].diff().fillna(0)
    DF['Daily_Log_Fut_Move']=np.log(DF['Close']).diff().fillna(0)
    return(DF)
def merge_Vol_and_Futures_DF(Vol_DF,Futures_DF):
    Merged_DFG=pd.concat([Vol_DF,Futures_DF], join='inner', axis=1)
    return(Merged_DFG)

def get_Quandl_Futures_History(exchange,Symbol):


    string_Quandl="CHRIS/"+exchange+"_"+Symbol+"1"
    print(string_Quandl)
    DF=quandl.get(string_Quandl)
    DF1=DF[['Open','High','Low','Last']]
    DF1.columns=['Open','High','Low','Close']
    return(DF1)
def plot_Quandl_Futures_Data(DF,title1):


    qf=cf.QuantFig(DF,title='CL',legend=title1,name=title1)
    qf.add_bollinger_bands()
    qf.iplot(kind='ohlc',up_color='blue',down_color='red')

def get_correlated_dataset(n, dependency, mu, scale):


    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):

    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

from scipy.stats import norm
from scipy.stats import mvn

def _gbs(option_type, fs, x, t, r, b, v):


   
    # -----------
    # Create preliminary calculations
    t__sqrt = math.sqrt(t)
    d1 = (math.log(fs / x) + (b + (v * v) / 2) * t) / (v * t__sqrt)
    d2 = d1 - v * t__sqrt

    if option_type == "c":
        # it's a call
       
        value = fs * math.exp((b - r) * t) * norm.cdf(d1) - x * math.exp(-r * t) * norm.cdf(d2)
        delta = math.exp((b - r) * t) * norm.cdf(d1)
        gamma = math.exp((b - r) * t) * norm.pdf(d1) / (fs * v * t__sqrt)
        theta = -(fs * v * math.exp((b - r) * t) * norm.pdf(d1)) / (2 * t__sqrt) - (b - r) * fs * math.exp(
            (b - r) * t) * norm.cdf(d1) - r * x * math.exp(-r * t) * norm.cdf(d2)
        vega = math.exp((b - r) * t) * fs * t__sqrt * norm.pdf(d1)
        rho = x * t * math.exp(-r * t) * norm.cdf(d2)
    else:
        # it's a put
        
        value = x * math.exp(-r * t) * norm.cdf(-d2) - (fs * math.exp((b - r) * t) * norm.cdf(-d1))
        delta = -math.exp((b - r) * t) * norm.cdf(-d1)
        gamma = math.exp((b - r) * t) * norm.pdf(d1) / (fs * v * t__sqrt)
        theta = -(fs * v * math.exp((b - r) * t) * norm.pdf(d1)) / (2 * t__sqrt) + (b - r) * fs * math.exp(
            (b - r) * t) * norm.cdf(-d1) + r * x * math.exp(-r * t) * norm.cdf(-d2)
        vega = math.exp((b - r) * t) * fs * t__sqrt * norm.pdf(d1)
        rho = -x * t * math.exp(-r * t) * norm.cdf(-d2)


    return value, delta, gamma, theta, vega, rho
#    fs          = price of underlying
#    x           = strike
#    t           = time to expiration
#    v           = implied volatility
#    r           = risk free rate
#    q           = dividend payment
#    b           = cost of carry

#    value       = price of the option
#    delta       = first derivative of value with respect to price of underlying
#    gamma       = second derivative of value w.r.t price of underlying
#    theta       = first derivative of value w.r.t. time to expiration
#    vega        = first derivative of value w.r.t. implied volatility
#    rho         = first derivative of value w.r.t. risk free rates

def black_76(option_type, fs, x, t, r, v):


    b = 0
    return _gbs(option_type, fs, x, t, r, b, v)

def custom_round(x, base=5):


    return int(base * round(float(x)/base))




def get_Strike_List(DF,strike_increment_div,strike_list_round,strike_list_div):


    strike_list=list(range(custom_round(int(DF['MinStrike']/strike_list_round),strike_increment_div), custom_round(int(DF['MaxStrike']/strike_list_round),strike_increment_div),strike_increment_div))
    futures_rounded=custom_round(int(DF['Future_Expand']/strike_list_round),strike_increment_div)/strike_list_div
    newList = map(lambda x: x/strike_list_div, strike_list)
    return(list(newList),futures_rounded)
def get_Strike_Matrix_Floating_Curve(round_shift,B2,B1):
    

    #B2=self.Strike_Details_DF
    B2['Shift']=B2['Vol'].shift(round_shift).fillna(B2.iloc[0]['Vol'])
    #B2['Shift']=B2['Shift']+round_shift*self.Strike_Matrix.iloc[-1]['SlopeValue']/10
    B2['Shift']=B2['Shift']+round_shift*B1.iloc[-1]['SlopeValue']/10
    return(B2)
#get_Strike_Matrix_Floating_Curve(1,B2,B1)   
#Daily[Daily['Strike']==custom_round(int(Original_Close*100),5)/100]['Slope'].values[0]
#Base_Day.iloc[i]['FuturesRound']
def calculate_Strike_MatrixAloneFeb18(DF,strike_increment_mult,strike_increment_div,strike_list_round,strike_list_div):
    

    #DF['StrikeMatrix']=0
    #DF['StrikeMatrix'].astype('object')
    DFStat=[]
    for i in range(len(DF)):
            
        #strike_list=list(range(int(DF.iloc[i]['MinStrike'])*strike_increment_mult, int(DF.iloc[i]['MaxStrike'])*strike_increment_mult,strike_increment_div))
        strike_list,futures_rounded=get_Strike_List(DF.iloc[i],strike_increment_div,strike_list_round,strike_list_div)
        #strike_list=list(range(custom_round(int(DF.iloc[i]['MinStrike']),strike_increment_div), custom_round(int(DF.iloc[i]['MaxStrike']),strike_increment_div),strike_increment_div))
        #print(strike_list)
        r=.0001
        q=0
        F=DF.iloc[i]['Future']
       # print(F)
        time=DF.iloc[i]['DtE']/365
        #print(t)
        strike_list_vol=[]
        for t in range(len(strike_list)):
            #print(strike_list[t])
            strike_vol=get_Vol_at_Strike(strike_list[t],DF.iloc[i])
            #print(strike_vol)
            #t=DF.iloc[i]['DtE']/365
            #value       = price of the option
            #    delta,gamma, theta, vega  ,  rho    = first derivative of value with respect to price of underlying
            #    gamma       = second derivative of value w.r.t price of underlying
            #    theta       = first derivative of value w.r.t. time to expiration
            #    vega        = first derivative of value w.r.t. implied volatility
            #    rho         = first derivative of value w.r.t. risk free rates  
            #print(F)
            #print(strike_list[t])
            #print(t)
            #print(r)
            #print(strike_vol)
            C_price, C_delta,C_gamma, C_theta, C_vega  ,  C_rho=black_76('c',F, strike_list[t], time, r, strike_vol)
            #print(C_price)
            #print(C_delta)
            P_price,P_delta,P_gamma, P_theta, P_vega  ,  P_rho=black_76('p',F, strike_list[t], time, r, strike_vol)
    
            #C_price = black_scholes_merton('c', S, strike_list[t], t, r, strike_vol, q)
            #C_delta_calc = analytical.delta('c', S, strike_list[t], t, r, strike_vol)
            #P_price = black_scholes_merton('p', S, strike_list[t], t, r, strike_vol, q)
            #P_delta_calc = analytical.delta('p', S, strike_list[t], t, r, strike_vol)
            strike_list_vol.append([strike_list[t],strike_vol,round(C_price,3),round(P_price,3),C_delta,P_delta])
            #print(strike_list_vol)
        DFStat.append([DF.iloc[i]['Future'],DF.iloc[i]['AtM'],DF.iloc[i]['DtE'],DF.iloc[i]['MinStrike'],DF.iloc[i]['MaxStrike'],strike_list_vol,futures_rounded] )
            #bbb.iloc[p, bbb.columns.get_loc('PnL')]
    DFStat_Df=pd.DataFrame(DFStat)
    return(DFStat_Df)
#calculate_Strike_MatrixAloneOneDayMove_New_Feb18(New.iloc[-1],Original_Close+round_shift,newVol,1,1,1,B2,B1,1,1)

def calculate_Strike_MatrixAloneOneDayMove_New_Feb18(DF,NewFuture,NewATM,strike_increment_mult,strike_increment_div,round_shift,B2,B1,strike_list_round,strike_list_div):


    # round shift is the change in futures price
    # strike_increment_mult is the 
    # strike_increment_div is the difference between strikes for strike list 
    
    #DF['StrikeMatrix']=0
    #DF['StrikeMatrix'].astype('object')
    DFStat=[]

            
    #strike_list=list(range(int(DF['MinStrike']), int(DF['MaxStrike']),strike_increment))
    #strike_list=list(range(int(DF.iloc[i]['MinStrike'])*strike_increment_mult, int(DF.iloc[i]['MaxStrike'])*strike_increment_mult,strike_increment_div))
    #strike_list=list(range(custom_round(int(DF.iloc[i]['MinStrike']),5), custom_round(int(DF.iloc[i]['MaxStrike']),5),strike_increment_div))
    strike_list,futures_rounded=get_Strike_List(DF,strike_increment_div,strike_list_round,strike_list_div)
    #strike_list=list(range(custom_round(int(DF['MinStrike']),5), custom_round(int(DF['MaxStrike']),5),strike_increment_div))
    #strike_vol=get_Vol_at_Strike_Shifted(strike_list[t]/strike_increment_mult,DF,NewFuture,NewATM)
    #print(strike_list)    
    r=.0001
    q=0
    F=NewFuture
    #print(F)
    time=(DF['DtE']-1)/365
        #print(t)
    strike_list_vol=[]
    B2=get_Strike_Matrix_Floating_Curve(round_shift,B2,B1)
    #C_price, C_delta,C_gamma, C_theta, C_vega  ,  C_rho=black_76('c',F, B2.iloc[t]['Strike'], time, r, strike_vol)
    
    
    for t in range(len(B2)):
        #print(strike_list[t])
        strike_vol=get_Vol_at_Strike_Shifted(strike_list[t],DF,NewFuture,NewATM)
        strike_vol=B2.iloc[t]['Shift']
        print(strike_vol)
            #t=DF.iloc[i]['DtE']/365
            #value       = price of the option
            #    delta,gamma, theta, vega  ,  rho    = first derivative of value with respect to price of underlying
            #    gamma       = second derivative of value w.r.t price of underlying
            #    theta       = first derivative of value w.r.t. time to expiration
            #    vega        = first derivative of value w.r.t. implied volatility
            #    rho         = first derivative of value w.r.t. risk free rates  
        #print(F)
        #print(strike_list[t])
        #print(t)
        #print(r)
        #print(strike_vol)
        C_price, C_delta,C_gamma, C_theta, C_vega  ,  C_rho=black_76('c',F, B2.iloc[t]['Strike'], time, r, strike_vol)
        #print(C_price)
        #print(C_delta)
        P_price,P_delta,P_gamma, P_theta, P_vega  ,  P_rho=black_76('p',F,B2.iloc[t]['Strike'] , time, r, strike_vol)
    
            #C_price = black_scholes_merton('c', S, strike_list[t], t, r, strike_vol, q)
            #C_delta_calc = analytical.delta('c', S, strike_list[t], t, r, strike_vol)
            #P_price = black_scholes_merton('p', S, strike_list[t], t, r, strike_vol, q)
            #P_delta_calc = analytical.delta('p', S, strike_list[t], t, r, strike_vol)
        strike_list_vol.append([B2.iloc[t]['Strike'],strike_vol,round(C_price,3),round(P_price,3),C_delta,P_delta])
            #print(strike_list_vol)
    DFStat.append([DF['Future'],DF['AtM'],DF['DtE'],DF['MinStrike'],DF['MaxStrike'],strike_list_vol,futures_rounded] )
            #bbb.iloc[p, bbb.columns.get_loc('PnL')]
    DFStat_Df=pd.DataFrame(DFStat)
    return(DFStat_Df)

def get_Vol_at_Strike(strike,DF):
    """
    Generate a dataframe which gives the PnL of each node of the binary tree for entry

    Parameters
    ----------
    df : pd.DataFrame
        a buy or sell dataframe of all the trades from the system

    Returns
    -------
    BinaryTree_DF : pd.DataFrame
        A dataframe with the PnL of trades at each binary tree node from a trade dataframe

    """

    x=np.log(strike/DF['Future'])
    Beta1=DF['Beta1']
    Beta2=DF['Beta2']
    Beta3=DF['Beta3']
    Beta4=DF['Beta4']
    Beta5=DF['Beta5']
    Beta6=DF['Beta5']
    AtM=DF['AtM']
#$D$5+$I$5*C22+$J$5*C22^2+$K$5*C22^3+$L$5*C22^4+$M$5*C22^5+$N$5*C22^6
    Vol_by_strike=AtM+Beta1*x+Beta2*x**2+Beta3*x**3+Beta4*x**4+Beta5*x**5+Beta6*x**6
    return(Vol_by_strike)
def get_Vol_at_Strike_Shifted(strike,DF,NewFuture,NewATM):
    """
    Generate a dataframe which gives the PnL of each node of the binary tree for entry

    Parameters
    ----------
    df : pd.DataFrame
        a buy or sell dataframe of all the trades from the system

    Returns
    -------
    BinaryTree_DF : pd.DataFrame
        A dataframe with the PnL of trades at each binary tree node from a trade dataframe

    """

    x=np.log(strike/NewFuture)
    Beta1=DF['Beta1']
    Beta2=DF['Beta2']
    Beta3=DF['Beta3']
    Beta4=DF['Beta4']
    Beta5=DF['Beta5']
    Beta6=DF['Beta5']
    AtM=NewATM
#$D$5+$I$5*C22+$J$5*C22^2+$K$5*C22^3+$L$5*C22^4+$M$5*C22^5+$N$5*C22^6
    Vol_by_strike=AtM+Beta1*x+Beta2*x**2+Beta3*x**3+Beta4*x**4+Beta5*x**5+Beta6*x**6
    return(Vol_by_strike)




class Contract_Name_Feb14:
    """
        A class used to represent an Single Commodity

        Sample call:
            UB=Contract_Name("ZB",1000,1,5,100,10,.03125,.0625,15)
        ...

        Attributes
        ----------
        Contract_Name : str
            the name of contract
        Tick_Size  : int
            the value of 1 point move
        Contract_Decimal_Places : float
            the multiplier to adjust decimal place
       length_of_MVA : int
            how many bars MVA for Signal
        length_of_Price : int
            how many bars MVA for Price for Trend
        smooth_factor : int
            how many bars for Hull MVA of signal. This smooths the choppy 5 bar to make uptick and downtick signals easier
        base_tick : float
            minimum tick value. used to calculate PnL
        target : float
            change from entry price to buy in Target strategy. If you are not filled on working bid price and you hit this a buy stop is activated to get long. PnL deducted accordingly
        short_term_price_MVA : int
           length of Hull MVA for MVA entry strategy. Strategy waits to buy until this MVA has positive slope or upticks

        Methods
        -------

        """

    def __init__(self, Contract_Name, Tick_Size, Contract_Decimal_Places, strike_spread,OW_Code,which_folder,strike_increment_div,strike_list_round,strike_list_div,price_bucket_levels):
        self.Contract_Name = Contract_Name
        self.Tick_Size = Tick_Size
        self.Contract_Decimal_Places = Contract_Decimal_Places

        self.strike_spread=strike_spread
        self.OW_Code = OW_Code
        self.which_folder = which_folder
        self.strike_increment_div=strike_increment_div
        self.strike_list_round=strike_list_round
        self.strike_list_div=strike_list_div
        self.price_bucket_levels=price_bucket_levels
        #self.roll_column = roll_column
        #self.the_hurdle_point = the_hurdle_point
        # roll_table = pd.read_csv(self.roll_table_csv)
    def Chart_Vol_vs_Price(self,DF,start_time_frame,end_time_frame,titleName,FuturesMove,VolMove):


        #DF=self.Term_Structure
        DF2=DF[start_time_frame :end_time_frame]

        #import plotly.graph_objs as go
        trace1 = go.Scatter(
        x=DF2[FuturesMove],
        y=DF2[VolMove],
        # Add a name for each trace to appear in the legend
        name = 'PriceLevel 1', 
        mode='markers',
        #marker=dict(color='rgba(152, 0, 0, .8)', size=4, showscale=False))
        #marker=dict(color='rgba(255, 182, 193, .9)', size=4, showscale=False))
        marker=dict(color='blue', size=4, showscale=False))


        #rgba(255, 182, 193, .9)
        # Join them all together 
        data = [trace1]

        layout = dict(
            title=titleName,
        xaxis=dict(title='Price Movement'),
        yaxis=dict(title='Vol Movement'))
        fig = dict(data=data, layout=layout)
        iplot(fig)
    
    def Plot_JointPlot(self,DF,start_time_frame ,end_time_frame,FuturesMove,VolMove):

        DF3=DF[start_time_frame :end_time_frame]
        
        sns.set(rc={'figure.figsize':(20.7,20.27)})
        ymin_Val=DF3[VolMove].min()*1.1
        ymax_Val=DF3[VolMove].max()*1.3
        xmin_Val=DF3[FuturesMove].min()*1.1
        xmax_Val=DF3[FuturesMove].max()*1.1
        sns.jointplot(data=DF3,
                  x=DF3[FuturesMove],
                  y=DF3[VolMove],
                  kind='kde',
                  space=2,height=15,
                  xlim=(xmin_Val,xmax_Val),
                  ylim=(ymin_Val,ymax_Val))
        
    def Chart_Ellipse_Standard_Deviations(self,DF,start_time_frame ,end_time_frame,FuturesMove,VolMove):


        DF=DF[start_time_frame :end_time_frame]
        fig, ax_nstd = plt.subplots(figsize=(16, 16))

        dependency_nstd = [[0.8, 0.75],
                   [-0.2, 0.35]]
        mu = 0, 0
        scale = 8, 5

        ax_nstd.axvline(c='grey', lw=1)
        ax_nstd.axhline(c='grey', lw=1)

        x, y = get_correlated_dataset(500, dependency_nstd, mu, scale)
        x=DF[FuturesMove]
        y=DF[VolMove]
        ax_nstd.scatter(x, y, s=15)

        confidence_ellipse(x, y, ax_nstd, n_std=1,
                   label=r'$1\sigma$', edgecolor='firebrick')
        confidence_ellipse(x, y, ax_nstd, n_std=2,
                   label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
        confidence_ellipse(x, y, ax_nstd, n_std=3,
                   label=r'$3\sigma$', edgecolor='blue', linestyle=':')

        ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
        ax_nstd.set_title('Standard deviations Zones')
        ax_nstd.legend()
        plt.show()
        
    def Chart_Histograms(self,DF,start_time_frame ,end_time_frame,FuturesMove,VolMove):


        DF=DF[start_time_frame :end_time_frame]
        DF[FuturesMove].iplot(kind='histogram',title=FuturesMove, bins=100)
        DF[VolMove].iplot(kind='histogram',title=VolMove, bins=100)
    def Create_Term_StructureOld(self):


        term_structure=["1W","1M","2M","3M","6M","1Y"]
    #base_Char="DF=pd.DataFrame(Data_List2[0])
        baseChar=self.OW_Code
        Data_List2=[]
        for i in range(len(term_structure)):
            string_code=baseChar+term_structure[i]+"_IVM"
            print(string_code)
            Data_List2.append(quandl.get(string_code))
        Data_List2[0]['1M']=Data_List2[1]['AtM']
        Data_List2[0]['2M']=Data_List2[2]['AtM']
        Data_List2[0]['3M']=Data_List2[3]['AtM']
        Data_List2[0]['6M']=Data_List2[4]['AtM']
        Data_List2[0]['1Y']=Data_List2[5]['AtM']
        DF=pd.DataFrame(Data_List2[0])
        
        DF['Daily_Futures_Move']=DF['Future'].diff().fillna(0)
        DF['Daily_Log_Futures_Move']=np.log(DF['Future']).diff().fillna(0)
        DF['Daily_Vol_Move']=DF['AtM'].diff().fillna(0)*100
        DF['Daily_Log_Vol_Move']=np.log(DF['AtM']).diff().fillna(0)*100

        DF['MeanDev']=DF['Future']*DF['AtM']*math.sqrt((1/365))
       
        DF['MaxStrike']=np.exp(DF['MaxMoney'])*DF['Future']*self.Contract_Decimal_Places
        DF['MinStrike']=np.exp(DF['MinMoney'])*DF['Future']*self.Contract_Decimal_Places
        DF['Future_Expand']=DF['Future']*self.Contract_Decimal_Places
        DF['MaxStrike']=DF['MaxStrike'].astype(int)
        DF['MinStrike']=DF['MinStrike'].astype(int)
        DF['Future_Expand']=DF['Future_Expand'].astype(int)
        

        DF['SecondDif']=DF['1M']-DF['2M']
        DF['FrontDif']=DF['AtM']-DF['1M']
        DF['MeanDev_1W']=DF['Future']*DF['AtM']*math.sqrt((1/365))
        DF['MeanDev_1M']=DF['Future']*DF['1M']*math.sqrt((1/365))
        DF['MeanDev_2M']=DF['Future']*DF['2M']*math.sqrt((1/365))
        DF['MeanDev_3M']=DF['Future']*DF['3M']*math.sqrt((1/365))
        DF['MeanDev_6M']=DF['Future']*DF['6M']*math.sqrt((1/365))
        DF['MeanDev_1Y']=DF['Future']*DF['1Y']*math.sqrt((1/365))

#int(math.exp( Crude.Term_Structure.iloc[-1]['MaxMoney'] )*Crude.Term_Structure.iloc[-1]['Future']),int(math.exp( Crude.Term_Structure.iloc[-1]['MinMoney'] )*Crude.Term_Structure.iloc[-1]['Future'])
        self.Term_Structure=DF
        return(DF)
        
    def Create_Term_Structure(self,Term):


        #term_structure=["1W","1M","2M","3M","6M","1Y"]
        #base_Char="DF=pd.DataFrame(Data_List2[0])
        baseChar=self.OW_Code
       
        string_code=baseChar+Term+"_IVM"
        print(string_code)
        DF=quandl.get(string_code)

        DF=self.Calc_Values(DF)
        #DF2=self.calculate_Strike_Matrix(DF)
        #self.DF2=DF2

#int(math.exp( Crude.Term_Structure.iloc[-1]['MaxMoney'] )*Crude.Term_Structure.iloc[-1]['Future']),int(math.exp( Crude.Term_Structure.iloc[-1]['MinMoney'] )*Crude.Term_Structure.iloc[-1]['Future'])
        self.Term_Structure_DF=DF
        return(DF)
    def Calc_Values(self,DF):


        DF['Daily_Futures_Move']=DF['Future'].diff().fillna(0)
        DF['Daily_Log_Futures_Move']=np.log(DF['Future']).diff().fillna(0)
        DF['Daily_Vol_Move']=DF['AtM'].diff().fillna(0)*100
        DF['Daily_Log_Vol_Move']=np.log(DF['AtM']).diff().fillna(0)*100

        DF['MeanDev']=DF['Future']*DF['AtM']*math.sqrt((1/365))
       
        DF['MaxStrike']=np.exp(DF['MaxMoney'])*DF['Future']*self.Contract_Decimal_Places
        DF['MinStrike']=np.exp(DF['MinMoney'])*DF['Future']*self.Contract_Decimal_Places
        DF['Future_Expand']=DF['Future']*self.Contract_Decimal_Places
        DF['MaxStrike']=DF['MaxStrike'].astype(int)
        DF['MinStrike']=DF['MinStrike'].astype(int)
        DF['Future_Expand']=DF['Future_Expand'].astype(int)
        
#int(math.exp( Crude.Term_Structure.iloc[-1]['MaxMoney'] )*Crude.Term_Structure.iloc[-1]['Future']),int(math.exp( Crude.Term_Structure.iloc[-1]['MinMoney'] )*Crude.Term_Structure.iloc[-1]['Future'])
        #self.Term_Structure_DF=DF
        return(DF)
    
    def Chart_OW_Data(self,DF,column_to_Chart,start_time_frame,end_time_frame):


        DF2=DF[start_time_frame :end_time_frame]
        titleIs=column_to_Chart+' History'
        DF2[column_to_Chart].iplot(title=titleIs)
        #DF2[column_to_Chart].iplot(title=titleIs,kind='bar')
        
    def Scatter_OW_Chart(self,DF,column_to_Chart,start_time_frame,end_time_frame):


        DF2=DF[start_time_frame :end_time_frame]
        DF2.iplot(kind="scatter",x="Future",y=column_to_Chart)
            
    def Scatter_OW_Chart_Markers(self,DF,column_to_Chart,start_time_frame,end_time_frame):


        DF2=DF[start_time_frame :end_time_frame]
        titleIs='Future versus '+column_to_Chart
        DF2.iplot(kind="scatter",x="Future",y=column_to_Chart, mode ='markers',title=titleIs) 
        
    def get_Quandl_Data(self,Exchange,Symbol,Term,Kind):



        Lookup_String="OWF/"+Exchange+"_"+Symbol+"_"+Symbol+"_"+Term+"_"+Kind
        print(Lookup_String)
        Data_Info=quandl.get(Lookup_String)
        Data_Info['Daily_Futures_Move']=Data_Info['Future'].diff().fillna(0)
        Data_Info['Daily_Log_Futures_Move']=np.log(Data_Info['Future']).diff().fillna(0)
        Data_Info['Daily_Vol_Move']=Data_Info['AtM'].diff().fillna(0)*100
        Data_Info['Daily_Log_Vol_Move']=np.log(Data_Info['AtM']).diff().fillna(0)*100
        Data_Info['Implied Move']=Data_Info['Future']*Data_Info['AtM']*(Data_Info['DtT']/365).pow(1./2)
        Data_Info['Mean Dev']=Data_Info['Future']*Data_Info['AtM']*math.sqrt((1/365))
        
        Data_Info['MaxStrike']=np.exp(Data_Info['MaxMoney'])*Data_Info['Future']*self.Contract_Decimal_Places
        Data_Info['MinStrike']=np.exp(Data_Info['MinMoney'])*Data_Info['Future']*self.Contract_Decimal_Places
        Data_Info['Future_Expand']=Data_Info['Future']*self.Contract_Decimal_Places
        Data_Info['MaxStrike']=Data_Info['MaxStrike'].astype(int)
        Data_Info['MinStrike']=Data_Info['MinStrike'].astype(int)
        Data_Info['Future_Expand']=Data_Info['Future_Expand'].astype(int)
        self.Data_History=Data_Info
        return(Data_Info)
    def get_Quandl_IV_Surface_Data(self,Exchange,Symbol,Term,Kind):


        Lookup_String="OWF/"+Exchange+"_"+Symbol+"_"+Symbol+"_"+Term+"_"+Kind
        print(Lookup_String)
        Data_Info=quandl.get(Lookup_String)
        
        self.IV_Surface_Data_History=Data_Info
        return(Data_Info)
    
    def get_Historical_Moves(self,DF,Stat,Biggest,number_to_get,cols_display):


        
        if Biggest==True:
            df_T=DF[cols_display].sort_values(by=[Stat],ascending=False).head(number_to_get)
            display(DF[cols_display].sort_values(by=[Stat],ascending=False).head(number_to_get))
            st.dataframe(df_T)
        else:
            df_T=DF[cols_display].sort_values(by=[Stat],ascending=True).head(number_to_get)
            display(DF[cols_display].sort_values(by=[Stat],ascending=True).head(number_to_get))
            st.dataframe(df_T)
            

            
    def get_Animation_Delta_Curve_DF(self,df):


        pdList=[]
        for i in range(len(df)):
    
            Strikes=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]

            strikevols=['P05dVol', 'P10dVol', 'P15dVol', 'P20dVol',
               'P25dVol', 'P30dVol', 'P35dVol', 'P40dVol', 'P45dVol', 'P50dVol',
               'P55dVol', 'P60dVol', 'P65dVol', 'P70dVol', 'P75dVol', 'P80dVol',
               'P85dVol', 'P90dVol', 'P95dVol']
            Vols=df.iloc[i][strikevols]
            Dates=[df.iloc[i]['Date']] * len(df)
            Dates
            df5=pd.DataFrame(list(zip(Strikes, Vols,Dates)), 
               columns =['Delta', 'Vold','Date'])
            pdList.append(df5)
            #px.line(df5, x="Delta", y="Vold")
        dfT=pd.concat(pdList)
        dfT['Date']=dfT['Date'].astype(str)
        dfT
        px.line(dfT, x="Delta", y="Vold", 
                 width=1200, height=800,animation_frame="Date",range_y=[min(dfT['Vold']),max(dfT['Vold'])])
        return(dfT)
    def get_Animation_Term_Structure_DF(self,df):


        pdList=[]
        for i in range(len(df)):
    
            Strikes=['1_W','1_M','2_M','3_M','6_M','1_Y']

            strikevols=['AtM', '1M', '2M', '3M',
               '6M', '1Y']
            Vols=df.iloc[i][strikevols]
            Dates=[df.iloc[i]['Date']] * len(df)
            Dates
            df5=pd.DataFrame(list(zip(Strikes, Vols,Dates)), 
               columns =['Delta', 'Vold','Date'])
            pdList.append(df5)
            #px.line(df5, x="Delta", y="Vold")
        dfT=pd.concat(pdList)
        dfT['Date']=dfT['Date'].astype(str)
        dfT
        fig10=px.line(dfT, x="Delta", y="Vold", 
                 width=1200, height=800,animation_frame="Date",range_y=[min(dfT['Vold']),max(dfT['Vold'])])
        st.header('Term Structure Animation')
        st.plotly_chart(fig10)
        return(dfT)
    
    def calculate_Strike_Matrix(self,DF):


        #DF['StrikeMatrix']=0
        #DF['StrikeMatrix'].astype('object')
        DFStat=[]
        for i in range(len(DF)):
            
            strike_list=list(range(int(DF.iloc[i]['MinStrike']), int(DF.iloc[i]['MaxStrike']),1))
    
            r=.001
            q=0
            S=DF.iloc[i]['Future']
            #print(S)
            t=DF.iloc[i]['DtE']/365
            #print(t)
            strike_list_vol=[]
            for t in range(len(strike_list)):
                #print(strike_list[i])
                strike_vol=get_Vol_at_Strike(strike_list[t],DF.iloc[i])
                #print(strike_vol)
                #t=DF.iloc[i]['DtE']/365
                C_price = black_scholes_merton('c', S, strike_list[t], t, r, strike_vol, q)
                C_delta_calc = analytical.delta('c', S, strike_list[t], t, r, strike_vol)
                P_price = black_scholes_merton('p', S, strike_list[t], t, r, strike_vol, q)
                P_delta_calc = analytical.delta('p', S, strike_list[t], t, r, strike_vol)
                strike_list_vol.append([strike_list[t],strike_vol,round(C_price,3),round(P_price,3),C_delta_calc,P_delta_calc])
            #print(strike_list_vol)
            DFStat.append([DF.iloc[i]['Future'],DF.iloc[i]['AtM'],DF.iloc[i]['DtE'],DF.iloc[i]['MinStrike'],DF.iloc[i]['MaxStrike'],strike_list_vol] )
            #bbb.iloc[p, bbb.columns.get_loc('PnL')]
        DFStat_Df=pd.DataFrame(DFStat)
        return(DFStat_Df)
        
    def get_Strike_Matrix_DF(self,DF):


        Base_Day=calculate_Strike_MatrixAlone(DF,self.strike_spread,self.strike_increment_div)
        Base_Day.columns=['Futures','AtM','DtE','MinS','MaxS','Strikes']
        Base_Day['FuturesRound']=Base_Day['Futures'].astype(int)
        #DF=Base_Day
        Base_Day['SlopeValue']=0
        for i in range(len(Base_Day)):
            ATM_Price=Base_Day.iloc[i]['FuturesRound']
    
            Daily=pd.DataFrame(Base_Day['Strikes'][i])
            Daily.columns=['Strike','Vol','Call','Put','C_Vol','P_Vol']
            Daily['Straddle']=Daily['Call']+Daily['Put']
            Daily['Slope']=Daily['C_Vol'].diff().fillna(0)
            Daily.loc[0,'Slope']=Daily.iloc[1]['Slope']
            slope_Value=Daily[Daily['Strike']==ATM_Price]['Slope'].values[0]
            Base_Day.loc[i,'SlopeValue']=slope_Value
        self.Strike_Matrix_DF=Base_Day
        self.Strike_Details=Daily
        return(Base_Day,Daily)
    def Create_Term_Structure_Single(self,Term_Selected):

        term_structure=["1W","1M","2M","3M","6M","1Y"]
        #base_Char="DF=pd.DataFrame(Data_List2[0])
        baseChar=self.OW_Code
        Data_List2=[]
        Base_List=[]
        for i in range(len(term_structure)):
            string_code=baseChar+term_structure[i]+"_IVM"
            print(string_code)
            Data_List2.append(quandl.get(string_code))
        index_num=term_structure.index(Term_Selected)
        string_code=baseChar+term_structure[index_num]+"_IVM"
        Base_List=quandl.get(string_code)
        DF=pd.DataFrame(Data_List2[0])
        DF['1W_V']=Data_List2[0]['AtM']
        DF['1M_V']=Data_List2[1]['AtM']
        DF['2M_V']=Data_List2[2]['AtM']
        DF['3M_V']=Data_List2[3]['AtM']
        DF['6M_V']=Data_List2[4]['AtM']
        DF['1Y_V']=Data_List2[5]['AtM']
        #DF=pd.DataFrame(Data_List2[0])
        
        DF['Daily_Futures_Move']=DF['Future'].diff().fillna(0)
        DF['Daily_Log_Futures_Move']=np.log(DF['Future']).diff().fillna(0)
        DF['Daily_Vol_Move']=DF['AtM'].diff().fillna(0)*100
        DF['Daily_Log_Vol_Move']=np.log(DF['AtM']).diff().fillna(0)*100

        DF['MeanDev']=DF['Future']*DF['AtM']*math.sqrt((1/365))
       
        DF['MaxStrike']=np.exp(DF['MaxMoney'])*DF['Future']*self.Contract_Decimal_Places
        DF['MinStrike']=np.exp(DF['MinMoney'])*DF['Future']*self.Contract_Decimal_Places
        DF['Future_Expand']=DF['Future']*self.Contract_Decimal_Places
        DF['MaxStrike']=DF['MaxStrike'].astype(int)
        DF['MinStrike']=DF['MinStrike'].astype(int)
        DF['Future_Expand']=DF['Future_Expand'].astype(int)

        DF['SecondDif']=DF['1M_V']-DF['2M_V']
        DF['FrontDif']=DF['1W_V']-DF['1M_V']
        DF['MeanDev_1W']=DF['Future']*DF['1W_V']*math.sqrt((1/365))
        DF['MeanDev_1M']=DF['Future']*DF['1M_V']*math.sqrt((1/365))
        DF['MeanDev_2M']=DF['Future']*DF['2M_V']*math.sqrt((1/365))
        DF['MeanDev_3M']=DF['Future']*DF['3M_V']*math.sqrt((1/365))
        DF['MeanDev_6M']=DF['Future']*DF['6M_V']*math.sqrt((1/365))
        DF['MeanDev_1Y']=DF['Future']*DF['1Y_V']*math.sqrt((1/365))

        #int(math.exp( Crude.Term_Structure.iloc[-1]['MaxMoney'] )*Crude.Term_Structure.iloc[-1]['Future']),int(math.exp( Crude.Term_Structure.iloc[-1]['MinMoney'] )*Crude.Term_Structure.iloc[-1]['Future'])
        self.Term_Structure_Single=DF
        return(DF)
    def get_Strike_Matrix_DF(self,DF):


        #Base_Day=calculate_Strike_MatrixAlone(DF,self.strike_spread,self.strike_increment_div)
        Base_Day=calculate_Strike_MatrixAloneFeb18(DF,self.strike_spread,self.strike_increment_div,self.strike_list_round,self.strike_list_div)
        Base_Day.columns=['Futures','AtM','DtE','MinS','MaxS','Strikes','Future_Expand2']
        Base_Day['FuturesRound']=Base_Day['Future_Expand2'].astype(int)
        #DF=Base_Day
        Base_Day['SlopeValue']=0
        for i in range(len(Base_Day)):
            ATM_Price=Base_Day.iloc[i]['FuturesRound']
    
            Daily=pd.DataFrame(Base_Day['Strikes'][i])
            Daily.columns=['Strike','Vol','Call','Put','C_Delta','P_Delta']
            Daily['Straddle']=Daily['Call']+Daily['Put']
            Daily['Slope']=Daily['Vol'].diff().fillna(0)
            Daily.loc[0,'Slope']=Daily.iloc[1]['Slope']
            slope_Value=Daily[Daily['Strike']==ATM_Price]['Slope'].values[0]
            Base_Day.loc[i,'SlopeValue']=slope_Value
        self.Strike_Matrix_DF=Base_Day
        self.Strike_Details=Daily
        return(Base_Day,Daily)
        
        
def Pick_Month(argument):
    switcher = {
        "Jan":'F',
        "Feb":'G',
        "Mar":'H',
        "Apr":'J',
        "May":'K',
        "Jun":'M',
        "Jul":'N',
        "Aug":'Q',
        "Sep":'U',
        "Oct":'V',
        "Nov":'X',
        "Dec":'Z'
    }
    return( switcher.get(argument, "Invalid month"))		

####################################################################################
#
# 	Declare Risk Slide Parameter List
#
##########################################################################################

CL_price_buckets=[-16,-4.8,-3,-1.6,0, 1.6, 3,4.8,16]
NG_price_buckets=[-1.8,-.54,-.36,-.18,0, .18,.36,.54,1.8]
GC_price_buckets=[-320,-96,-64,-32,0, 32,64,96,320]
ZF_price_buckets=[-1.72,-.52,-.34,-.17,0, .17,.34,.52,1.72]
ZB_price_buckets=[-15.63,-4.69,-3.13,-1.56,0, 1.56,3.13,4.69,15.63]
ZN_price_buckets=[-4.69,-1.41,-.94,-.47,0, .47,.94,1.41,4.69]


####  Declare Classes #####



Crude=Contract_Name_Feb14("Crude",10,1,1,"OWF/NYM_CL_CL_",'C:/Users/mpgen/OptionWorks/Crude',1,1,1,CL_price_buckets)
NaturalGas=Contract_Name_Feb14("NAturalGas",10,1000,1000,"OWF/NYM_NG_NG_",'C:/Users/mpgen/OptionWorks/NaturalGas',5,10,100,NG_price_buckets)
Gold=Contract_Name_Feb14("Gold",10,1,1000,"OWF/CMX_GC_GC_",'C:/Users/mpgen/OptionWorks/Gold',5,1,1,GC_price_buckets)
TenYear=Contract_Name_Feb14("TenYear",10,1,1000,"OWF/CBT_TY_TY_",'C:/Users/mpgen/OptionWorks/TenYear',5,1,1,ZN_price_buckets)
FiveYear=Contract_Name_Feb14("FiveYear",10,1,1000,"OWF/CBT_FV_FV_",'C:/Users/mpgen/OptionWorks/FiveYear',5,1,1,ZF_price_buckets)
ThirtyYear=Contract_Name_Feb14("ThirtyYear",10,1,1000,"OWF/CBT_US_US_",'C:/Users/mpgen/OptionWorks/ThirtyYear',5,1,1,ZB_price_buckets)

####    Read in Continous Futures Files    ####


Gold_Futures = pd.read_csv("GCJ21Current.txt")
Crude_Futures = pd.read_csv("CLJ21Current.txt")
NaturalGas_Futures = pd.read_csv("NGJ21Current.txt")
ZB_Futures = pd.read_csv("ZBH21Current.txt")
ZN_Futures = pd.read_csv("ZNM21Current.txt")
ZF_Futures = pd.read_csv("ZFM21Current.txt")

Gold_Futures=transform_Text_Futures_File(Gold_Futures)
Crude_Futures=transform_Text_Futures_File(Crude_Futures)
NaturalGas_Futures=transform_Text_Futures_File(NaturalGas_Futures)
ZB_Futures=transform_Text_Futures_File(ZB_Futures)
ZN_Futures=transform_Text_Futures_File(ZN_Futures)
ZF_Futures=transform_Text_Futures_File(ZF_Futures)





############################



st.title('Historical Data dashboard')

#### Sidebar Creation #######


from datetime import datetime,date,time
add_selectbox = st.sidebar.header("Date Range Picker")
add_selectbox_start =st.sidebar.date_input('start date')
add_selectbox_finish =st.sidebar.date_input('end_date')
add_selectbox = st.sidebar.header("Select Underlying")
Which_Contract_options = st.sidebar.selectbox('Select Contract',['Crude', 'Natural Gas','Gold','10 Year','5 Year','30 Year'])
add_selectbox = st.sidebar.header("Constant or Individual")
Which_Structure_options = st.sidebar.selectbox('Select Structure',['Term Structure', 'Individual'])
if 'Individual' in Which_Structure_options:
    Month_options = st.sidebar.selectbox('Select Month',['Jan', 'Feb', 'Mar', 'Apr','May', 'Jun', 'Jul', 'Aug','Sep', 'Oct', 'Nov', 'Dec'])
    Year_options = st.sidebar.selectbox('Select Year',['2009', '2010', '2011', '2012','2013', '2014', '2015', '2016','2017', '2018', '2019', '2020','2021'])
    Other_Chart_Values_options = st.sidebar.multiselect('Chart Values',['AtM', 'Mean Dev','RR25','RR10'])

if 'Term Structure' in Which_Structure_options:
    Term_options = st.sidebar.selectbox('Select Month',['1W', '1M', '2M', '3M','6M', '1Y'])
    Other_Chart_Values_options = st.sidebar.multiselect('Chart Values',[ 'AtM','SecondDif','MeanDev_1W','MeanDev_1M','MeanDev_2M','MeanDev_3M','MeanDev_6M','MeanDev_1Y','RR25','RR10'])
    #DF_Term=class_Selection.Create_Term_Structure_Single(Term_options)

add_selectbox = st.sidebar.header("Chart Types")
Chart_Type_options = st.sidebar.multiselect('Chart Types',['Histogram', 'Line', 'Clouds', 'Path','JointPlot', 'Ellipse Distribution','Delta Curve Animation','Bar Chart'])
Chart_Values_options = st.sidebar.selectbox('Futures Values',['Daily_Fut_Move', 'Daily_Fut_Log_Move'])
Table_Values_options = st.sidebar.selectbox('Vol Values',['Daily_Vol_Move', 'Daily_Vol_Log_Move'])


    
    
add_selectbox = st.sidebar.header("Moves Table")
Show_Table_Values_options = st.sidebar.selectbox('Order',['Yes', 'No'])
Show_Table_Values = st.sidebar.selectbox('Chart Scale',['Daily_Vol_Move', 'Daily_Vol_Log_Move','Daily_Fut_Move', 'Daily_Fut_Log_Move'])
Total_Table_Values_options = st.sidebar.selectbox('Top Number',['5', '10'])

genre = st.sidebar.radio("Display Strike Grid?",('Yes', 'No'))
if genre == 'Yes':
    #st.write('You selected comedy.')
    add_selectbox_sheets = st.sidebar.header("Strike Date Range Picker")
    add_selectbox_start_sheets =st.sidebar.date_input('Strike start date')
    add_selectbox_finish_sheets =st.sidebar.date_input('Strike end_date')
Byrn_slides = st.sidebar.radio("Display Byrn_Slides?",('Yes', 'No'))

############################







if st.button('Run'):
    #st.write('Why hello there')
    if 'Crude' in Which_Contract_options:
        contract_abbreviation= "CL"
        exchange_name='NYM'
        class_Selection =Crude
        class_Selection.FuturesData=Crude_Futures
    if 'Natural Gas' in Which_Contract_options:
        contract_abbreviation= "NG"
        exchange_name='NYM'
        class_Selection =NaturalGas
        class_Selection.FuturesData=NaturalGas_Futures
    if 'Gold' in Which_Contract_options:
        contract_abbreviation= "GC"
        class_Selection =Gold
        exchange_name='CMX'
        class_Selection.FuturesData=Gold_Futures
    if '10 Year' in Which_Contract_options:
        contract_abbreviation= "TY"
        class_Selection =TenYear
        exchange_name='CBT'
        class_Selection.FuturesData=ZN_Futures
    if '5 Year' in Which_Contract_options:
        contract_abbreviation= "FV"
        class_Selection =FiveYear
        exchange_name='CBT'
        class_Selection.FuturesData=ZF_Futures
    if '30 Year' in Which_Contract_options:
        contract_abbreviation= "US"
        class_Selection =ThirtyYear
        exchange_name='CBT'
        class_Selection.FuturesData=ZB_Futures

    if 'Term Structure' in Which_Structure_options:
        DF_Term=class_Selection.Create_Term_Structure_Single(Term_options)
        DF=DF_Term
    else:
        Contract_Code=Pick_Month(Month_options)+Year_options
        class_Selection.get_Quandl_Data(exchange_name,contract_abbreviation,Contract_Code,"IVM")
        DF=class_Selection.Data_History
    class_Selection.Create_Term_StructureOld()
    
#
#    Define parameters from sidebar
#

    start_time_frame=add_selectbox_start
    end_time_frame=add_selectbox_finish
    

    FuturesMove=Chart_Values_options

    VolMove=Table_Values_options

    column_to_Chart=Other_Chart_Values_options
    DF=DF[start_time_frame :end_time_frame]
    
    Cont_Contract_Data=get_Quandl_Futures_History("CME",contract_abbreviation)
    DF=merge_Vol_and_Futures_DF(DF,class_Selection.FuturesData)
    DF_Term_Futures=merge_Vol_and_Futures_DF(class_Selection.Term_Structure,class_Selection.FuturesData)
    
#
#    Execute Selected Options
#
#

    if 'Histogram' in Chart_Type_options:
        #fig =DF[FuturesMove].iplot(kind='histogram',title=FuturesMove, bins=100,asFigure=True)
        fig =DF[FuturesMove].iplot(kind='histogram',title=FuturesMove, bins=100,asFigure=True)
        fig2 =DF[VolMove].iplot(kind='histogram',title=VolMove, bins=100,asFigure=True)
        st.header('Price Histogram')
        st.plotly_chart(fig)
        st.header('Vol Histogram')
        st.plotly_chart(fig2)

        
    if 'Line' in Chart_Type_options:
        #fig3=DF[column_to_Chart].iplot(title=titleIs,asFigure=True)
        st.header('Chart History')
        get_Line_Chart(DF,Other_Chart_Values_options)
        #st.plotly_chart(fig3) 

    if 'Clouds' in Chart_Type_options:
        #fig5=DF.iplot(kind="scatter",x="Future",y=column_to_Chart, mode ='markers',title="Cloud History",asFigure=True)

        st.header('Cloud History')
        get_Cloud_Chart(DF,Other_Chart_Values_options)

        #st.plotly_chart(fig5) 
    if 'Path' in Chart_Type_options:
        #fig4=DF.iplot(kind="scatter",x="Future",y=column_to_Chart,asFigure=True)
        
        st.header('Vol Path History')
        get_Path_Chart(DF,Other_Chart_Values_options)
        #st.plotly_chart(fig4)
        
    if 'JointPlot' in Chart_Type_options:
        sns.set(rc={'figure.figsize':(20.7,20.27)})
        ymin_Val=DF[VolMove].min()*1.1
        ymax_Val=DF[VolMove].max()*1.3
        xmin_Val=DF[FuturesMove].min()*1.1
        xmax_Val=DF[FuturesMove].max()*1.1
        fig6=sns.jointplot(data=DF,
                  x=DF[FuturesMove],
                  y=DF[VolMove],
                  kind='kde',
                  space=2,height=15,
                  xlim=(xmin_Val,xmax_Val),
                  ylim=(ymin_Val,ymax_Val))
        st.header('Joint Plot')
        st.pyplot(fig6)
    if 'Ellipse Distribution' in Chart_Type_options:
        fig7, ax_nstd = plt.subplots(figsize=(16, 16))

        dependency_nstd = [[0.8, 0.75],
                   [-0.2, 0.35]]
        mu = 0, 0
        scale = 8, 5

        ax_nstd.axvline(c='grey', lw=1)
        ax_nstd.axhline(c='grey', lw=1)

        x, y = get_correlated_dataset(500, dependency_nstd, mu, scale)
        x=DF[FuturesMove]
        y=DF[VolMove]
        ax_nstd.scatter(x, y, s=15)

        confidence_ellipse(x, y, ax_nstd, n_std=1,
                   label=r'$1\sigma$', edgecolor='firebrick')
        confidence_ellipse(x, y, ax_nstd, n_std=2,
                   label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
        confidence_ellipse(x, y, ax_nstd, n_std=3,
                   label=r'$3\sigma$', edgecolor='blue', linestyle=':')

        ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
        ax_nstd.set_title('Standard deviations Zones')
        ax_nstd.legend()
        plt.show()
        st.header('Standard Deviation Ellipse Distribution')
        st.pyplot(fig7)
    if 'Delta Curve Animation' in Chart_Type_options:
        if 'Individual' in Which_Structure_options:
            df=class_Selection.get_Quandl_IV_Surface_Data(exchange_name,contract_abbreviation,Contract_Code,"IVS")
        else:
            df=class_Selection.get_Quandl_IV_Surface_Data(exchange_name,contract_abbreviation,Term_options,"IVS")
        #df=Crude.get_Quandl_IV_Surface_Data("NYM","CL","J2021","IVS")
        df2=df[start_time_frame:end_time_frame]
        df2.reset_index(inplace=True)   
    
        
        pdList=[]

        for i in range(len(df2)):
    
            Strikes=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]

            strikevols=['P05dVol', 'P10dVol', 'P15dVol', 'P20dVol',
               'P25dVol', 'P30dVol', 'P35dVol', 'P40dVol', 'P45dVol', 'P50dVol',
               'P55dVol', 'P60dVol', 'P65dVol', 'P70dVol', 'P75dVol', 'P80dVol',
               'P85dVol', 'P90dVol', 'P95dVol']
            Vols=df2.iloc[i][strikevols]
            Dates=[df2.iloc[i]['Date']] * len(df)
            #Dates
            df5=pd.DataFrame(list(zip(Strikes, Vols,Dates)), 
               columns =['Delta', 'Vold','Date'])
            pdList.append(df5)
            #px.line(df5, x="Delta", y="Vold")
        dfT=pd.concat(pdList)
        dfT['Date']=dfT['Date'].astype(str)
        #dfT
        fig9=px.line(dfT, x="Delta", y="Vold", 
                 width=1200, height=800,animation_frame="Date",range_y=[min(dfT['Vold']),max(dfT['Vold'])])

        st.header('Surface Animation')
        st.plotly_chart(fig9)
        
        df2=class_Selection.Term_Structure[start_time_frame:end_time_frame]
        df2.reset_index(inplace=True)
        dfTT=Crude.get_Animation_Term_Structure_DF(df2)
        
        
    if 'Bar Chart' in Chart_Type_options:
        layout1 = cf.Layout(height=1000,width=1000)
        qf=class_Selection.FuturesData[start_time_frame :end_time_frame]
        fig10=qf.iplot(kind='ohlc',up_color='blue',down_color='red',asFigure=True,layout=layout1)
        st.header('OHLC Chart')
        st.plotly_chart(fig10)
        #st.plotly_chart(qf)
        #qf.add_bollinger_bands()
        
        #plot_Quandl_Futures_Data(Cont_Contract_Data[start_time_frame :end_time_frame],contract_abbreviation)
    #if 'Yes' in Show_Table_Values_options:
        #Show_Table_Values_values
    cols_display=['Future','AtM','Daily_Futures_Move','Daily_Log_Futures_Move','Daily_Fut_Move','Daily_Log_Fut_Move','Daily_Vol_Move','Daily_Log_Vol_Move','SecondDif','FrontDif','MeanDev_1W','MeanDev_1M','MeanDev_2M','MeanDev_3M','MeanDev_6M','MeanDev_1Y']

    st.header('Biggest Moves')
    if 'Yes' in Show_Table_Values_options:
        class_Selection.get_Historical_Moves(DF_Term_Futures[start_time_frame :end_time_frame],Show_Table_Values,True,int(Total_Table_Values_options),cols_display)
        class_Selection.get_Historical_Moves(DF_Term_Futures[start_time_frame :end_time_frame],Show_Table_Values,False,int(Total_Table_Values_options),cols_display)
    #else:
        #class_Selection.get_Historical_Moves(DF_Term_Futures[start_time_frame :end_time_frame],Show_Table_Values,False,int(Total_Table_Values_options),cols_display)
    
    if 'Yes' in genre:
        st.header('Strike Matrix')
        start_time_frame_sheets=add_selectbox_start_sheets
        end_time_frame_sheets=add_selectbox_finish_sheets
        New=DF[start_time_frame_sheets :end_time_frame_sheets]
        B1,B2=class_Selection.get_Strike_Matrix_DF(New)
        st.dataframe(B2)
    if Byrn_slides == 'Yes':
        st.header('Byrnhildr Slide Matrix')
        DF3=get_Bin_Distribution(DF,class_Selection.price_bucket_levels)
        st.dataframe(DF3)
        

    
        
        
        
else:
	st.write('Goodbye')
    
	
#streamlit run
