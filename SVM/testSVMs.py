#coding: utf-8
from pymprog import *
begin('bike production')
x, y = var('x, y') # 变量
maximize(15 * x + 10 * y, 'profit') # 目标函数
x <= 3 # 约束条件1
y <= 4 # 约束条件2
x + y <= 5 # 约束条件3
solve()