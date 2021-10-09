from sympy import Symbol
from sympy import solve
from sympy import diff
from fractions import Fraction

x=Symbol('x')
y=Symbol('y')
t=Symbol('t')

def unconstrain_opt(function,point,epsilon):
    ### Function That Optimize Unconstrain Problems Through Gradiant Descent And Exact Line Search
    ### Function: Objective Function We Want To Minimize
    ### Point: Initial point
    ### Epsilon: Threshold to stop the search
    iter=1
    stop_value=99999999
    ### Take The Gradient
    gradient=[diff(function,x),diff(function,y)]

    while stop_value>epsilon:
        print("Iteration: "+str(iter))
        ### Evaluate The Gradient 
        gradient_eval=[gradient[0].subs({x:point[0],y:point[1]}),gradient[1].subs({x:point[0],y:point[1]})]
        ### Distance Between The Actual Point And The Gradiant
        distance=[i-(j)*t for i,j in zip(point,gradient_eval)]
        ### Evalute The Function In The New Variables
        new_func=function.subs({x:distance[0],y:distance[1]})
        ### Derivate The Function In T 
        new_func_dev=diff(new_func,t)
        ### Obtain optimal t
        t_val=solve(new_func_dev,t)[0]
        print('The Value Of T Is: '+str(Fraction(float(t_val))))
        ### Update The New Point 
        point=[i-(j*t_val) for i,j in zip(point,gradient_eval)]
        print('The Value Of X1 Is: '+str(point[0]))
        print('The Value Of X2 Is: '+str(point[1]))
        
        ### Stop Criterion 
        stop=[gradient[0].subs({x:point[0],y:point[1]}),gradient[1].subs({x:point[0],y:point[1]})]
        ### Stop Value
        stop_value=sum([i**2 for i in stop])**(1/2)
        print('The Stop Criteria Is: '+str(Fraction(float(stop_value))))
        iter=iter+1
        ### Objetive Function 
        fo=function.subs({x:point[0],y:point[1]})
        print('The Objective Funtion: '+str(float(fo)))

unconstrain_opt(x**2-2*x*y+2*(y**2)+2*x,[0,0],0.01)


