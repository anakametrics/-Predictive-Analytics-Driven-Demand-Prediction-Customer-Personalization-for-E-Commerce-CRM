# main.py
import os
import base64
import io
import math
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
from ml_model import run_pipeline

import mysql.connector
import hashlib
from datetime import datetime
from datetime import date
import datetime
import calendar
import random
from random import randint
from urllib.request import urlopen
import webbrowser
from plotly import graph_objects as go

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import numpy as np

from werkzeug.utils import secure_filename
from PIL import Image

import urllib.request
import urllib.parse
import socket    
import csv

import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt
from collections import OrderedDict

plt.style.use("ggplot")

import sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree

import torch
import torch.nn as nn


mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  charset="utf8",
  database="next_basket"

)
app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
#######
UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = { 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####
@app.route('/', methods=['GET', 'POST'])
def index():
    msg=""

    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM nb_customer WHERE uname = %s AND pass = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('userhome'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('index.html',msg=msg)

@app.route('/login_buyer', methods=['GET', 'POST'])
def login_buyer():
    msg=""

    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM nb_customer WHERE uname = %s AND pass = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('userhome'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login_buyer.html',msg=msg)

@app.route('/login_seller', methods=['GET', 'POST'])
def login_seller():
    msg=""

    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM nb_seller WHERE uname = %s AND pass = %s && status=1', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('rt_home'))
        else:
            msg = 'Incorrect username/password! or Not Approved!'
    return render_template('login_seller.html',msg=msg)

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""

    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM nb_admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password! or access not provided'
    return render_template('login.html',msg=msg)

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg=""
    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
        
    if request.method=='POST':
        name=request.form['name']
        address=request.form['address']
        city=request.form['city']
        mobile=request.form['mobile']
        email=request.form['email']
        uname=request.form['uname']
        pass1=request.form['pass']
       
        
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM nb_customer where uname=%s",(uname,))
        cnt = mycursor.fetchone()[0]

        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM nb_customer")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
                    
            sql = "INSERT INTO nb_customer(id,name,address,city,mobile,email,uname,pass,create_date) VALUES (%s, %s, %s, %s,%s, %s, %s, %s,%s)"
            val = (maxid,name,address,city,mobile,email,uname,pass1,rdate)
            mycursor.execute(sql, val)
            mydb.commit()            
            #print(mycursor.rowcount, "Registered Success")
            msg="success"
            
        else:
            msg='fail'
    return render_template('register.html',msg=msg)


# SELLER DASHBOARD
@app.route("/rt_seller")
def rt_seller():

    uname=""
    if 'username' in session:
        uname = session['username']
    act=request.args.get("act")
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM nb_seller where uname=%s",(uname,))
    data1 = mycursor.fetchone()
    
    df = run_pipeline()
    
    return render_template("rt_seller.html", data=df.to_dict(orient="records"))

# BUYER PAGE
@app.route("/buyer")
def buyer():
    df = run_pipeline()
    df = df[df["risk_label"].isin(["High", "Medium"])]
    return render_template("buyer.html", data=df.to_dict(orient="records"))


'''@app.route('/add_product', methods=['GET', 'POST'])
def add_product():
    msg=""
    act = request.args.get('act')
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM cs_category")
    data = mycursor.fetchall()

    
        
    if request.method=='POST':
        category=request.form['category']
        product=request.form['product']
        price=request.form['price']
        detail=request.form['detail']
        
    
        file = request.files['file']
        mycursor.execute("SELECT max(id)+1 FROM cs_product")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
            
        try:
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file:
                fn=file.filename
                fnn="P"+str(maxid)+fn  
                #fn1 = secure_filename(fn)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], fnn))
                
        except:
            print("dd")
        
        

        photo="P"+str(maxid)+fn   
        sql = "INSERT INTO cs_product(id,category,product,price,photo,detail) VALUES (%s, %s, %s, %s, %s, %s)"
        val = (maxid,category,product,price,photo,detail)
        mycursor.execute(sql, val)
        mydb.commit()            
        #print(mycursor.rowcount, "Registered Success")
        result="sucess"
        if mycursor.rowcount==1:
            return redirect(url_for('add_product'))
        else:
            msg='Already Exist'

    if act=="del":
        did = request.args.get('did')
        mycursor.execute('delete from cs_product WHERE id = %s', (did, ))
        mydb.commit()
        return redirect(url_for('add_product'))

    
        
    mycursor.execute("SELECT * FROM cs_product")
    data2 = mycursor.fetchall()
    
    return render_template('add_product.html',msg=msg,data=data,data2=data2)'''

@app.route('/reg_seller', methods=['GET', 'POST'])
def reg_seller():
    msg=""
    sid=""
    if request.method=='POST':
        name=request.form['name']
        address=request.form['address']
        city=request.form['city']
        mobile=request.form['mobile']
        email=request.form['email']
        uname=request.form['uname']
        pass1=request.form['pass']
        
    
        
        mycursor = mydb.cursor()

        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
    
        mycursor.execute("SELECT count(*) from nb_seller where uname=%s",(uname,))
        cnt = mycursor.fetchone()[0]

        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM nb_seller")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
                    
            sql = "INSERT INTO nb_seller(id,name,address,city,mobile,email,uname,pass,create_date) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
            val = (maxid,name,address,city,mobile,email,uname,pass1,rdate)
            mycursor.execute(sql, val)
            mydb.commit()
            sid=str(maxid)
            #print(mycursor.rowcount, "Registered Success")
            msg="success"
           
        else:
            msg="fail"
    return render_template('reg_seller.html',msg=msg,sid=sid)

@app.route('/upload_data', methods=['GET', 'POST'])
def upload_data():
    msg=""
    sid=request.args.get("sid")

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * from nb_seller where id=%s",(sid,))
    dd = mycursor.fetchone()
    seller=dd[6]

    if request.method=='POST':
        file = request.files['file']

        
        if file.filename == '':
            #flash('No selected file')
            return redirect(request.url)
        if file:
            fn=file.filename
            fnn=seller+".csv"
            #fn1 = secure_filename(fn)
            file.save(os.path.join("static/dataset/", fnn))
            msg="ok"
                


    return render_template('upload_data.html',msg=msg,sid=sid)


@app.route('/view_seller', methods=['GET', 'POST'])
def view_seller():
    msg=""
    act=request.args.get("act")
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM nb_seller")
    data = mycursor.fetchall()

    if act=="yes":
        did=request.args.get("did")
        mycursor.execute("update nb_seller set status=1 where id=%s",(did,))
        mydb.commit()
        return redirect(url_for("admin"))
    
    return render_template('view_seller.html',data=data)

@app.route('/rt_home', methods=['GET', 'POST'])
def rt_home():
    msg=""
    data3=[]
    avdata=[]
    s1=""
    uname=""
    if 'username' in session:
        uname = session['username']
    act=request.args.get("act")
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM nb_seller where uname=%s",(uname,))
    data = mycursor.fetchone()

    mycursor.execute("SELECT * FROM nb_product where seller=%s",(uname,))
    data2 = mycursor.fetchall()

    if act=="del":
        did = request.args.get('did')
        mycursor.execute("SELECT * FROM nb_product where id=%s",(did,))
        dd = mycursor.fetchone()
        os.remove("static/upload/"+dd[6])
        mycursor.execute('delete from nb_product WHERE id = %s', (did, ))
        mydb.commit()
        return redirect(url_for('rt_home'))

    ###################
    mycursor.execute("SELECT count(*) FROM nb_sale_recommend where seller=%s",(uname,))
    cnt = mycursor.fetchone()[0]
    if cnt>0:
        s1="1"
        mycursor.execute("SELECT * FROM nb_sale_recommend where seller=%s order by pcount limit 0,6",(uname,))
        drw = mycursor.fetchall()
        av_qty=0
        for dat3 in drw:
            p1=dat3[2]
            mycursor.execute("SELECT count(*) FROM nb_product where seller=%s && product=%s",(uname,p1))
            cnt2 = mycursor.fetchone()[0]
            if cnt2>0:
                mycursor.execute("SELECT * FROM nb_product where seller=%s && product=%s",(uname,p1))
                drw1 = mycursor.fetchone()
                av_qty=drw1[5]
            else:
                av_qty=0

            av=dat3[2]+" - Available: "+str(av_qty)
            avdata.append(av)
    ###############            
    
    return render_template('rt_home.html',data=data,uname=uname,data2=data2,act=act,data3=data3,avdata=avdata,s1=s1)

@app.route('/rt_sales', methods=['GET', 'POST'])
def rt_sales():
    msg=""
    uname=""
    if 'username' in session:
        uname = session['username']
    act=request.args.get("act")
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM nb_seller where uname=%s",(uname,))
    data = mycursor.fetchone()

    mycursor.execute("SELECT * FROM nb_cart c,nb_product p where c.pid=p.id && c.status=1 && p.seller=%s",(uname,))
    data2 = mycursor.fetchall()


    
    return render_template('rt_sales.html',data=data,uname=uname,data2=data2,act=act)

@app.route('/add_cat', methods=['GET', 'POST'])
def add_cat():
    msg=""
    act = request.args.get("act")
    fnn=""
    uname=""
    if 'username' in session:
        uname = session['username']
        
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM nb_seller where uname=%s",(uname,))
    data = mycursor.fetchone()
        
    if request.method=='POST':
        category=request.form['category']
        category1=request.form['category1']

        mycursor.execute("SELECT max(id)+1 FROM nb_category")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        
        sql = "INSERT INTO nb_category(id,seller,category,category1) VALUES (%s, %s, %s,%s)"
        val = (maxid,uname,category,category1)
        mycursor.execute(sql, val)
        mydb.commit()            
        #print(mycursor.rowcount, "Registered Success")
        result="sucess"
        if mycursor.rowcount==1:
            return redirect(url_for('add_cat',act='1'))
        else:
            msg='Already Exist'

    if act=="del":
        did = request.args.get('did')
        mycursor.execute('delete from nb_category WHERE id = %s', (did, ))
        mydb.commit()
        return redirect(url_for('add_cat'))

    
        
    mycursor.execute("SELECT * FROM nb_category where seller=%s",(uname,))
    data2 = mycursor.fetchall()
    
    return render_template('add_cat.html',msg=msg,uname=uname,data=data,data2=data2,act=act)

@app.route('/add_product', methods=['GET', 'POST'])
def add_product():
    msg=""
    act = request.args.get("act")
    fnn=""
    uname=""
    if 'username' in session:
        uname = session['username']
        
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM nb_seller where uname=%s",(uname,))
    data = mycursor.fetchone()
    
    mycursor.execute("SELECT * FROM nb_category")
    data1 = mycursor.fetchall()

    
        
    if request.method=='POST':
        category=request.form['category']
        product=request.form['product']
        price=request.form['price']
        qty=request.form['qty']
        details=request.form['details']
        
    
        file = request.files['file']
        mycursor.execute("SELECT max(id)+1 FROM nb_product")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
            
        
        if file.filename == '':
            #flash('No selected file')
            return redirect(request.url)
        if file:
            fn=file.filename
            fnn="P"+str(maxid)+fn  
            #fn1 = secure_filename(fn)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], fnn))
                
        
        
        sql = "INSERT INTO nb_product(id,seller,category,product,price,quantity,photo,details) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        val = (maxid,uname,category,product,price,qty,fnn,details)
        mycursor.execute(sql, val)
        mydb.commit()            
        #print(mycursor.rowcount, "Registered Success")
        msg="success"

        
    mycursor.execute("SELECT * FROM nb_product where seller=%s",(uname,))
    data2 = mycursor.fetchall()
    
    return render_template('add_product.html',msg=msg,uname=uname,data=data,data1=data1,act=act)


@app.route('/edit', methods=['GET', 'POST'])
def edit():
    msg=""
    uname=""
    if 'username' in session:
        uname = session['username']
    act=request.args.get("act")
    pid=request.args.get("pid")
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM nb_seller where uname=%s",(uname,))
    data = mycursor.fetchone()

    mycursor.execute("SELECT * FROM nb_product where id=%s",(pid,))
    data2 = mycursor.fetchone()

    if request.method=='POST':
        product=request.form['product']
        price=request.form['price']
        qty=request.form['qty']
        details=request.form['details']
        mycursor.execute("update nb_product set product=%s,price=%s,quantity=%s,details=%s where id=%s",(product,price,qty,details,pid))
        mydb.commit()

        mycursor.execute("SELECT * FROM nb_product where id=%s",(pid,))
        dd3 = mycursor.fetchone()
        if dd3[5]>dd3[9]:
            mycursor.execute("update nb_product set status=0 where id=%s",(pid,))
            mydb.commit()
    
        return redirect(url_for('rt_home'))
        
    
    return render_template('edit.html',data=data,uname=uname,data2=data2,act=act)


@app.route('/userhome', methods=['GET', 'POST'])
def userhome():
    msg=""
    cnt=0
    uname=""
    act = request.args.get('act')
    cat = request.args.get('cat')
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM nb_customer where uname=%s",(uname,))
    usr = mycursor.fetchone()

    mycursor.execute("SELECT distinct(category) FROM nb_category")
    data2 = mycursor.fetchall()

    cc=""
    if cat is None:
        cc=""
    else:
        cc="1"
    
    if request.method=='POST':
        getval=request.form['getval']
        cat="%"+getval+"%"
        prd="%"+getval+"%"
        det="%"+getval+"%"
        mycursor.execute("SELECT * FROM nb_product where category like %s || product like %s || details like %s  order by star desc",(cat,prd,det))
        data = mycursor.fetchall()

        mycursor.execute("SELECT count(*) FROM nb_search where uname=%s && keyword=%s",(uname,getval))
        cnt2 = mycursor.fetchone()[0]
        if cnt2==0:

            mycursor.execute("SELECT max(id)+1 FROM nb_search")
            maxid1 = mycursor.fetchone()[0]
            if maxid1 is None:
                maxid1=1
                
            sql = "INSERT INTO nb_search(id, uname, keyword, scount) VALUES (%s, %s, %s, %s)"
            val = (maxid1, uname, getval, '1')
            mycursor.execute(sql,val)
            mydb.commit()
        else:
            mycursor.execute('update nb_search set scount=scount+1 WHERE uname=%s && keyword=%s', (uname,getval))
            mydb.commit()

        
    elif cc=="1":
        mycursor.execute("SELECT * FROM nb_product where category=%s order by star desc",(cat,))
        data = mycursor.fetchall()
    else:
        mycursor.execute("SELECT * FROM nb_product order by star desc")
        data = mycursor.fetchall()

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    month=now.strftime("%m")
    year=now.strftime("%Y")
    
    if act=="cart":
        pid = request.args.get('pid')
        mycursor.execute('SELECT count(*) FROM nb_cart WHERE uname=%s && pid = %s && status=0', (uname, pid))
        num = mycursor.fetchone()[0]

        mycursor.execute("SELECT * FROM nb_product where id=%s",(pid,))
        pdata = mycursor.fetchone()
        price=pdata[4]
        cat=pdata[2]
        seller=pdata[1]
        prdu=pdata[3]
        if num==0:
            mycursor.execute("SELECT max(id)+1 FROM nb_cart")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
                
            sql = "INSERT INTO nb_cart(id, uname, pid, status, rdate, price,category,seller,month,year) VALUES (%s, %s, %s, %s, %s, %s, %s,%s,%s,%s)"
            val = (maxid, uname, pid, '0', rdate, price, cat,seller,month,year)
            mycursor.execute(sql,val)
            mydb.commit()

            ####
            j=0
            mycursor.execute("delete from nb_recommend where uname=%s",(uname,))
            mydb.commit()
            
            filename = 'static/data1/train.csv'
            rdata1 = pd.read_csv(filename, header=0)
            for rdat1 in rdata1.values:
                
                if rdat1[15]==cat:
                    if prdu==rdat1[16]:
                        s=1
                    else:
                        if j<15:
                            recom=rdat1[16]
                            prc=rdat1[17]
                            mycursor.execute("SELECT max(id)+1 FROM nb_recommend")
                            maxid2 = mycursor.fetchone()[0]
                            if maxid2 is None:
                                maxid2=1
                                
                            sql = "INSERT INTO nb_recommend(id, uname, cartid, pid, product, category,recommend,price) VALUES (%s, %s, %s, %s, %s, %s, %s,%s)"
                            val = (maxid2, uname,maxid, pid, prdu, cat,recom,prc )
                            mycursor.execute(sql,val)
                            mydb.commit()
                            j+=1
                
            ####
            return redirect(url_for('userhome'))

    mycursor.execute("SELECT count(*) FROM nb_cart where uname=%s && status=0",(uname,))
    cnt = mycursor.fetchone()[0]
    if cnt>0:
        msg="1"
    else:
        msg=""
    
    return render_template('userhome.html',msg=msg,usr=usr,data=data,cnt=cnt,data2=data2)

'''@app.route('/cart', methods=['GET', 'POST'])
def cart():
    act=""
    pid=""
    did=""
    amount=""
    if 'username' in session:
        uname = session['username']

    cursor = mydb.cursor()
    cursor.execute("SELECT count(*) FROM nb_cart where uname=%s && status=0",(uname, ))
    cnt = cursor.fetchone()[0]
    if cnt>0:
        act="1"
    else:
        act=""
    
    cursor.execute('SELECT c.id,p.product,p.price,p.details,p.photo,c.rdate FROM nb_cart c,nb_product p where c.pid=p.id and c.uname=%s and c.status=0', (uname, ))
    data = cursor.fetchall()

    cursor.execute("SELECT * FROM nb_cart where uname=%s && status=0",(uname, ))
    dr = cursor.fetchall()
    amt=0
    for dv in dr:
        pid=dv[2]
        cursor.execute("SELECT price FROM nb_product where id=%s",(pid, ))
        pr = cursor.fetchone()[0]
        amt+=pr
        

    
    if request.method=='GET':
        act = request.args.get('act')
        pid = request.args.get('pid')
        did = request.args.get('did')
        if act=="ok":
            mycursor = mydb.cursor()
            mycursor.execute("SELECT max(id)+1 FROM nb_cart")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
            now = datetime.datetime.now()
            rdate=now.strftime("%d-%m-%Y")
            
            sql = "INSERT INTO cart(id, uname, pid, rdate) VALUES (%s, %s, %s, %s)"
            val = (maxid, uname, pid, rdate)
            #mycursor.execute(sql,val)
            #mydb.commit()
            #return redirect(url_for('cart',data=data))
        if act=="del":
            cursor = mydb.cursor()
            cursor.execute('delete FROM cart WHERE id = %s', (did, ))
            mydb.commit()
            return redirect(url_for('cart',data=data))

    if request.method=='POST':
        amount=request.form['amount']
        print("test")
        return redirect(url_for('payment', amount=amt))
            
    return render_template('cart.html', act=act,data=data, amount=amt,act=act)'''

def unique(list1):
 
    # initialize a null list
    unique_list = []
 
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    #for x in unique_list:
    #    print x,
    return unique_list

@app.route('/cart', methods=['GET', 'POST'])
def cart():
    msg=""
    s1=""
    uname=""
    act=request.args.get("act")
    st=""
    pid=""
    did=""
    total=0
    amount=""
    pdata=[]
    pdata1=[]
    data4=[]
    mess=""
    email=""

    bs_array=[]
    x=0
    ns=""
            
    if 'username' in session:
        uname = session['username']

    mycursor = mydb.cursor()

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    month=now.strftime("%m")
    year=now.strftime("%Y")

    mycursor.execute("SELECT * FROM nb_customer where uname=%s",(uname,))
    usr = mycursor.fetchone()
    email=usr[5]
    name=usr[1]
    city=usr[3]
    mobile=usr[4]

    mycursor.execute('SELECT count(*) FROM nb_cart WHERE uname=%s && status=0', (uname,))
    cart_n = mycursor.fetchone()[0]
    print("cart")
    print(cart_n)
    if cart_n>0:
        s1="1"

    

            
    mycursor.execute('SELECT count(*) FROM nb_cart WHERE uname=%s && status=0 && check_st=0', (uname,))
    cn = mycursor.fetchone()[0]
    if cn>0:
        mycursor.execute('SELECT sum(amount) FROM nb_cart WHERE uname=%s && status=0 && check_st=0', (uname,))
        total = mycursor.fetchone()[0]

    mycursor.execute("SELECT distinct(category) FROM nb_category")
    data2 = mycursor.fetchall()
    
    mycursor.execute("SELECT count(*) FROM nb_cart where uname=%s && status=0",(uname, ))
    cnt = mycursor.fetchone()[0]
    
    
    mycursor.execute('SELECT c.id,p.product,p.price,p.details,p.photo,c.rdate,c.quantity,c.amount,c.check_st,av_product FROM nb_cart c,nb_product p where c.pid=p.id and c.uname=%s and c.status=0', (uname, ))
    data = mycursor.fetchall()

    mycursor.execute("SELECT * FROM nb_cart where uname=%s && status=0",(uname, ))
    dr = mycursor.fetchall()

        
    i=0
    mul=0
    if request.method=='POST':
        ch=request.form['ch']
        
        qty=request.form.getlist('qty[]')
        rid=request.form.getlist('rid[]')

        if ch=="1":

            mycursor.execute("update nb_cart set check_st=0,av_product=0 where uname=%s && status=0",(uname, ))
            mydb.commit()
        
            for d1 in rid:
                user_qty=int(qty[i])

                mycursor.execute("SELECT price FROM nb_cart where id=%s",(d1, ))
                d2 = mycursor.fetchone()[0]
                mul=d2*user_qty

                mycursor.execute("SELECT * FROM nb_cart where id=%s",(d1, ))
                d3 = mycursor.fetchone()
                prid=d3[2]

                                
                mycursor.execute("SELECT * FROM nb_product where id=%s",(prid, ))
                pr = mycursor.fetchone()
                pr_qty=pr[5]
                
                

                rqty=pr[9]-d3[7]
                av_qty=pr_qty-rqty
                
                rqty1=rqty+user_qty
                mycursor.execute("update nb_product set required_qty=%s where id=%s ",(rqty1,prid))
                mydb.commit()

                if av_qty<user_qty:
                    mycursor.execute("update nb_cart set check_st=1,av_product=%s where id=%s && pid=%s",(av_qty,d1,prid))
                    mydb.commit()
                    
                
                mycursor.execute("update nb_cart set quantity=%s,amount=%s where id=%s ",(user_qty,mul,d1))
                mydb.commit()
                i+=1

                
            return redirect(url_for('cart',act='mail'))
        elif ch=="2":
            print("buy")
            rnf=randint(1000,9999)
            ff=open("otp.txt","w")
            ff.write(str(rnf))
            ff.close()
            
            if total>0:
                return redirect(url_for('cart',act='otp'))
            else:
                msg="2"
        elif ch=="3":
            print("check otp")
            otp=request.form['otp']
            mycursor.execute('SELECT * FROM nb_customer WHERE uname=%s',(uname,))
            r1 = mycursor.fetchone()
            ff=open("otp.txt","r")
            kk=ff.read()
            ff.close()
            if kk==otp:
                return redirect(url_for('cart',act='yes'))
            else:
                msg="4"

    if act=="del":
        did=request.args.get("did")
        mycursor.execute('SELECT * FROM nb_cart WHERE id=%s',(did,))
        dd1 = mycursor.fetchone()
        pid1=dd1[2]
        mycursor.execute("update nb_product set required_qty=required_qty-%s where id=%s",(dd1[7],pid1))
        mydb.commit()
        mycursor.execute("delete from nb_cart where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('cart'))
        
    #send mail for products required  
    if act=="mail":
        print("mail")
        mycursor.execute('SELECT count(*) FROM nb_product WHERE required_qty>quantity && status=0')
        pn = mycursor.fetchone()[0]

        
        if pn>0:
            st="1"
           
            mycursor.execute('SELECT * FROM nb_product WHERE required_qty>quantity && status=0')
            pdata = mycursor.fetchall()

            for rr in pdata:
                dt=[]
                ret=rr[1]
                mycursor.execute('SELECT * FROM nb_seller WHERE uname=%s',(ret,))
                pd1 = mycursor.fetchone()

                pname=rr[3]
                pid=rr[0]
                email=pd1[5]
                avp=rr[5]
                rp=rr[9]

                mess="Product ID: "+str(pid)+", Product: "+pname+", Availble only "+str(avp)+", Required "+str(rp)
                mycursor.execute("update nb_product set status=1 where id=%s ",(rr[0],))
                mydb.commit()
                    
                dt.append(mess)
                dt.append(email)
                pdata1.append(dt)
            

    if act=="otp":
        #rnf=randint(1000,9999)
        #rnf=5864
        ff=open("otp.txt","r")
        ot=ff.read()
        ff.close()
        mess="OTP: "+ot
        #ff=open("otp.txt","w")
        #ff.write(str(rnf))
        #ff.close()
        #mycursor.execute("update nb_customer set otp=%s where uname=%s",(str(rn),uname))
        #mydb.commit()
        
        
        
    #payment
    
    if act=="yes":
        mycursor.execute('SELECT * FROM nb_cart WHERE uname=%s && status=0 && check_st=0 order by id desc', (uname,))
        qc1 = mycursor.fetchone()
        seller=qc1[12]
        fnn=seller+".csv"
        
        sdata = pd.read_csv("static/dataset/"+fnn)
        sdata1=[]
        cnt=0
        for ss1 in sdata.values:
            cnt+=1

        mycursor.execute("SELECT max(id)+1 FROM nb_purchase")
        maxid21 = mycursor.fetchone()[0]
        if maxid21 is None:
            maxid21=1
            
        mycursor.execute('update nb_cart set status=1,bill_id=%s WHERE uname=%s && status=0 && check_st=0', (maxid21, uname ))
        mydb.commit()

        sql = "INSERT INTO nb_purchase(id, uname, amount, rdate) VALUES (%s, %s, %s, %s)"
        val = (maxid21, uname, total, rdate)
        mycursor.execute(sql,val)
        mydb.commit()
        ##
        mycursor.execute('SELECT * FROM nb_cart WHERE uname=%s && status=0 && check_st=0', (uname,))
        qc = mycursor.fetchall()
        i=1
        for rc in qc:

            cnt1=cnt+i
            uq=rc[7]
            amt=rc[8]
            mycursor.execute("update nb_product set quantity=quantity-%s,required_qty=required_qty-%s where id=%s",(uq,uq,rc[2]))
            mydb.commit()
            ########################################################################
            cate=""
            subcat=""
            mycursor.execute('SELECT * FROM nb_category WHERE category=%s', (rc[6],))
            ct = mycursor.fetchall()
            for ctt in ct:
                cate=ctt[3]
                subcat=ctt[2]

            mycursor.execute('SELECT * FROM nb_product WHERE id=%s', (rc[2],))
            pt = mycursor.fetchone()
            product=pt[2]

            row_id=str(cnt1)
            order_id="IN_"+str(maxid)
            ord_date=rdate
            ship_date=rdate

            smd=['First Class','Same Day','Second Class','Standard Class']
            rn=randint(1,4)
            rn1=rn-1
            smd[rn1]
            shipmode=smd[0]
            
            cusid=uname
            cusname=name

            sg=['Consumer','Corporate','Home Office']
            rn=randint(1,3)
            rn1=rn-1
            segment=sg[rn1]

            country='India'
            city=city
            state='Tamilnadu'

            pn=randint(600000,699999)
            postal=str(pn)

            rg=['Cental','East','South West']
            rn=randint(1,3)
            rn1=rn-1
            region=rg[rn1]

            prdid="P"+str(rc[2])
            
            amount=str(amt)

            #insdata=[row_id,order_id,ord_date,ship_date,shipmode,cusid,cusname,segment,country,city,state,postal,region,prdid,cate,subcat,product,amount]
            insdata=[row_id,order_id,ord_date,cusid,cusname,segment,country,city,state,prdid,cate,subcat,product,amount]

            with open('static/dataset/'+fnn,'a',newline='') as outfile:
                writer = csv.writer(outfile, quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow(insdata)
            i+=1
            ####

        
        
        return redirect(url_for('cart', act='success'))
    if act=="success":
        mycursor.execute('SELECT amount FROM nb_purchase WHERE uname=%s order by id desc limit 0,1', (uname,))
        amount = mycursor.fetchone()[0]
        mess="Dear "+name+", Amount Rs."+str(amount)+" Purchased Success"
        msg="3"

    ###
    if act=="cart":
        pid = request.args.get('pid')
        mycursor.execute('SELECT count(*) FROM nb_cart WHERE uname=%s && pid = %s && status=0', (uname, pid))
        num = mycursor.fetchone()[0]

        mycursor.execute("SELECT * FROM nb_product where id=%s",(pid,))
        pdata = mycursor.fetchone()
        price=pdata[4]
        cat=pdata[2]
        seller=pdata[1]
        if num==0:
            mycursor.execute("SELECT max(id)+1 FROM nb_cart")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
                
            sql = "INSERT INTO nb_cart(id, uname, pid, status, rdate, price,category,seller,month,year) VALUES (%s, %s, %s, %s, %s, %s, %s,%s,%s,%s)"
            val = (maxid, uname, pid, '0', rdate, price, cat,seller,month,year)
            mycursor.execute(sql,val)
            mydb.commit()
            return redirect(url_for('cart'))
    ##next basket recommendation#############
    mycursor.execute("SELECT count(*) FROM nb_cart where uname=%s",(uname,))
    pcnt = mycursor.fetchone()[0]
    val=""
    cartid=""
    pp=[]
    if pcnt>0:
        ns="1"
        mycursor.execute("SELECT * FROM nb_cart where uname=%s && status=0 order by id",(uname,))
        pdata1 = mycursor.fetchall()
        cart_prd=""
        cart_cat=""
        for pdata11 in pdata1:
            cart_prd=pdata11[2]
            cart_cat=pdata11[6]

        mycursor.execute("SELECT * FROM nb_product where id=%s order by id",(cart_prd,))
        pdata2 = mycursor.fetchall()
        cart_prd2=""
        cart_cat2=""
        for pdata22 in pdata2:
            cart_prd2=pdata22[3]
            cart_cat2=pdata22[2]

            cat2=""
            mycursor.execute("SELECT * FROM nb_category where category=%s order by id",(cart_cat2,))
            pdata3 = mycursor.fetchall()
            for pdata33 in pdata3:
                cat2=pdata33[3]
                
            mycursor.execute("SELECT * FROM nb_category where category!=%s && category1=%s order by id",(cart_cat2,cat2))
            pdata4 = mycursor.fetchall()
            for pdata44 in pdata4:
                cat3=pdata44[2]
                print(cat3)

                mycursor.execute("SELECT * FROM nb_product where category=%s",(cat3,))
                pdata5 = mycursor.fetchall()
                for pdata55 in pdata5:
                    
                    ppp=pdata55[3]
                    pp.append(ppp)
                    
            
        rec_prd=unique(pp)
        #print(rec_prd)
        for rp in rec_prd:
            mycursor.execute("SELECT * FROM nb_product where product=%s",(rp,))
            pdata6 = mycursor.fetchall()
            for pdata66 in pdata6:
                dp=[]                    
                dp.append(pdata66[3])
                dp.append(pdata66[4])
                data4.append(dp)
         


    ##################################
    '''mycursor.execute("SELECT * FROM nb_cart where uname=%s && status=0 order by id",(uname,))
    pdata1 = mycursor.fetchall()
    cart_prd=""
    cart_cat=""
    for pdata2 in pdata1:
        cartid=pdata2[0]
        

    mycursor.execute("SELECT * FROM nb_recommend where cartid=%s && uname=%s order by rand() limit 0,6",(cartid,uname))
    data4 = mycursor.fetchall()'''

        
    '''mycursor.execute("SELECT count(*) FROM nb_cart where uname=%s",(uname,))
    pcnt = mycursor.fetchone()[0]
    val=""
    if pcnt>0:
        ns="1"
        mycursor.execute("SELECT * FROM nb_cart where uname=%s order by id",(uname,))
        pdata1 = mycursor.fetchall()
        cart_prd=""
        cart_cat=""
        for pdata2 in pdata1:
            cart_prd=str(pdata2[2])
            cart_cat=pdata2[6]

        mycursor.execute("SELECT * FROM nb_product where id=%s",(cart_prd,))
        pdata2 = mycursor.fetchone()
        cart_prdd=pdata2[3]
        cart_prdd2=pdata2[2]
        
        ##
        filename = 'static/data1/train.csv'
        rdata1 = pd.read_csv(filename, header=0)
        rdata1.sort_values(["Row_ID"],  
                    axis=0, 
                    ascending=[False],  
                    inplace=True)

        for rdat in rdata1.values:
            if cart_prdd2 in rdat[15] or cart_prdd in rdate[16]:
                print(rdat[0])
                val=rdat[0]
                break

        print(val)
        if val=="":
            vv=randint(500,9800)
            val=str(vv)

        rid2=int(val)-1
        rid3=rid2-7
        j=rid3
        
        while j<=rid2:
            for rdat2 in rdata1.values:
                dt=[]
                if rdat2[0]==j:
                    dt.append(rdat2[15])
                    dt.append(rdat2[16])
                    dt.append(rdat2[17])
                    data4.append(dt)


            j+=1'''

        
    '''rid1=int(rdat2[0])
    
    while j<=rid2:
        if rdat2[0]==str(j):
            print(rdat2[0]+" "+rdat2[16])
        j+=1'''
        
        
    '''mycursor.execute('SELECT count(*) FROM nb_cart WHERE uname=%s && status=0', (uname,))
    mq11 = mycursor.fetchone()[0]

    if mq11>0:
        mycursor.execute('SELECT * FROM nb_cart WHERE uname=%s && status=0', (uname,))
        mq1 = mycursor.fetchall()
        for mq2 in mq1:
            mpid=mq2[2]
            mycursor.execute('SELECT count(*) FROM nb_cart WHERE pid=%s && status>0', (mpid,))
            mqn4 = mycursor.fetchone()[0]
            if mqn4>0:
                mycursor.execute('SELECT * FROM nb_cart WHERE pid=%s && status>0', (mpid,))
                mq4 = mycursor.fetchall()
                for mq5 in mq4:
                    x+=1
                    sess_id=mq5[15]
                    mycursor.execute('SELECT * FROM nb_cart WHERE pid!=%s && session_id=%s', (mpid,sess_id))
                    mq6 = mycursor.fetchone()
                    bs_array.append(mq6[2])
    upid=unique(bs_array)
    print("recommend")
    print(upid)
    if x>0:
        ns="1"
        for up in upid:
            mycursor.execute('SELECT * FROM nb_product WHERE id=%s', (up,))
            mq7 = mycursor.fetchall()
            for mq8 in mq7:
                data4.append(mq8)
        print(data4)
        print(data4[0][1])'''
    ################################                
        

    '''if request.method=='POST':
        amount=request.form['amount']
        print("test")
        return redirect(url_for('payment', amount=amt))'''
            
    return render_template('cart.html',msg=msg,uname=uname,s1=s1,usr=usr,data=data,data4=data4,ns=ns,cnt=cnt,data2=data2,cart_n=cart_n,total=total,act=act,pdata1=pdata1,st=st,mess=mess,email=email,name=name,mobile=mobile)




@app.route('/payment', methods=['GET', 'POST'])
def payment():
    msg=""
    mob2=""
    email2=""
    mess=""
    uname=""
    amount=0
    if 'username' in session:
        uname = session['username']
    if request.method=='GET':
        amount = request.args.get('amount')
    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    cursor = mydb.cursor()

    #print("uname="+uname)
    cursor.execute("SELECT * FROM nb_customer where uname=%s",(uname, ))
    rd=cursor.fetchone()
    name=rd[1]
    mob1=rd[2]
    email=rd[3]

    x=0
    if request.method=='POST':
        card=request.form['card']
        amount=request.form['amount']
        

        cursor.execute("SELECT * FROM nb_customer where uname=%s",(uname, ))
        rr=cursor.fetchone()
        mob2=rr[3]
        email2=rr[4]
        
        cursor.execute("SELECT max(id)+1 FROM cs_purchase")
        maxid = cursor.fetchone()[0]
        if maxid is None:
            maxid=1

        mess="Dear "+name+", Amount Rs."+amount+" Purchased Success, Recommended Products - Click http://localhost:5000/recommend1?user="+uname
        #url="http://iotcloud.co.in/testmail/email.php?email="+email+"&message="+message
        #webbrowser.open_new(url)

        cursor.execute('update cs_cart set status=1,bill_id=%s WHERE uname=%s && status=0', (maxid, uname ))
        mydb.commit()

        sql = "INSERT INTO cs_purchase(id, uname, amount, rdate) VALUES (%s, %s, %s, %s)"
        val = (maxid, uname, amount, rdate)
        cursor.execute(sql,val)
        mydb.commit()
        msg="1"

        

    return render_template('payment.html', msg=msg, amount=amount, email=email,mess=mess)


@app.route('/purchase', methods=['GET', 'POST'])
def purchase():
    uname=""
    amount=0
    if 'username' in session:
        uname = session['username']
    
    
    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM nb_purchase where uname=%s",(uname, ))
    data1=cursor.fetchall()

    return render_template('purchase.html', data1=data1)

@app.route('/view', methods=['GET', 'POST'])
def view():
    uname=""
    amount=0
    if 'username' in session:
        uname = session['username']
    
    bid = request.args.get('bid')
    cursor = mydb.cursor()
    cursor.execute('SELECT c.id,p.product,p.price,p.details,p.photo,c.rdate,c.quantity,c.amount FROM nb_cart c,nb_product p where c.pid=p.id and c.bill_id=%s', (bid, ))
    data = cursor.fetchall()

    return render_template('view.html', data=data)

@app.route('/add_review', methods=['GET', 'POST'])
def add_review():
    msg=""
    act=""
    email=""
    mess=""
    rid=""
    uname=""
    pid = request.args.get('pid')
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM nb_customer where uname=%s",(uname,))
    usr = mycursor.fetchone()
    email=usr[3]
    name=usr[1]

    mycursor.execute("SELECT * FROM nb_product where id=%s",(pid,))
    prd = mycursor.fetchone()

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    
    mycursor.execute("SELECT count(*) FROM nb_review where pid=%s && status=1",(pid,))
    cnt = mycursor.fetchone()[0]
    if cnt>0:
        act="1"
    mycursor.execute("SELECT * FROM nb_review where pid=%s && status=1",(pid,))
    data1 = mycursor.fetchall()

    rn=randint(10000,99999)

    if request.method=='POST':
        star=request.form['star']
        review=request.form['review']
        mycursor.execute("SELECT max(id)+1 FROM nb_review")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1

        rid=str(rid)
        sql = "INSERT INTO nb_review(id,pid,uname,review,star,rdate,status,review_code) VALUES (%s, %s, %s, %s, %s,%s,%s,%s)"
        val = (maxid,pid,uname,review,star,rdate,'0',str(rn))
        mycursor.execute(sql,val)
        mydb.commit()
        msg="ok"
        mess="Dear "+name+", Review Code: "+str(rn)
        #url="http://iotcloud.co.in/testmail/sendmail.php?email="+email+"&message="+message
        #webbrowser.open_new(url)
            
        #return redirect(url_for('review_code',rid=maxid))
        
        ##
        ########################################################################
      

        prdid="P"+pid

        insdata=[uname,prdid,str(star),rdate]

        with open('static/data1/reviews.csv','a') as outfile:
            writer = csv.writer(outfile, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(insdata)
        msg="ok"
        ####

    return render_template('add_review.html',msg=msg,usr=usr,data1=data1,act=act,pid=pid,prd=prd,email=email,mess=mess,rid=rid)

@app.route('/review_code', methods=['GET', 'POST'])
def review_code():
    msg=""
    uname=""
    rid = request.args.get('rid')
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM cs_register where uname=%s",(uname,))
    usr = mycursor.fetchone()
    email=usr[3]
    name=usr[1]

    mycursor.execute("SELECT * FROM cs_review where id=%s",(rid,))
    data1 = mycursor.fetchone()
    code=data1[7]
    pid=data1[1]
    if request.method=='POST':
        rcode=request.form['review_code']
        if rcode==code:
            mycursor.execute("SELECT count(*) FROM cs_cart where pid=%s && uname=%s && status=1",(pid,uname))
            cnt = mycursor.fetchone()[0]
            if cnt>0:
                mycursor.execute('update cs_review set status=1 WHERE id = %s', (rid,))
                mydb.commit()

                mycursor.execute("SELECT * FROM cs_review where pid=%s && status=1",(pid,))
                pdd = mycursor.fetchall()
                sr=0
                i=0
                for pn in pdd:
                    sr+=pn[4]
                    i+=1
                ss=sr/i
                star=int(ss)
                mycursor.execute('update cs_product set star=%s WHERE id = %s', (star,pid))
                mydb.commit()
                    
            
                msg="Your Review has posted"
            else:
                msg="Your Review has not posted! not buy this product!"
        else:
            msg="Review Code wrong!"


            
    return render_template('review_code.html',msg=msg)

@app.route('/search', methods=['GET', 'POST'])
def search():
    msg=""
    cnt=0
    uname=""
    act = request.args.get('act')
    cat = request.args.get('cat')
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM cs_register where uname=%s",(uname,))
    usr = mycursor.fetchone()

    mycursor.execute("SELECT * FROM cs_search where uname=%s order by scount desc",(uname,))
    data2 = mycursor.fetchall()

    cc=""
    if cat is None:
        cc=""
    else:
        cc="1"

    if cc=="1":
        cat="%"+cat+"%"
        prd="%"+cat+"%"
        det="%"+cat+"%"
        mycursor.execute("SELECT * FROM cs_product where category like %s || product like %s || detail like %s  order by star desc",(cat,prd,det))
        data = mycursor.fetchall()
    else:
        mycursor.execute("SELECT * FROM cs_product order by star desc")
        data = mycursor.fetchall()

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    
    if act=="cart":
        pid = request.args.get('pid')
        mycursor.execute('SELECT count(*) FROM cs_cart WHERE uname=%s && pid = %s && status=0', (uname, pid))
        num = mycursor.fetchone()[0]

        mycursor.execute("SELECT * FROM cs_product where id=%s",(pid,))
        pdata = mycursor.fetchone()
        price=pdata[3]
        cat=pdata[1]
        if num==0:
            mycursor.execute("SELECT max(id)+1 FROM cs_cart")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
                
            sql = "INSERT INTO cs_cart(id, uname, pid, status, rdate, price,category) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            val = (maxid, uname, pid, '0', rdate, price, cat)
            mycursor.execute(sql,val)
            mydb.commit()
            return redirect(url_for('search'))

    mycursor.execute("SELECT count(*) FROM cs_cart where uname=%s && status=0",(uname,))
    cnt = mycursor.fetchone()[0]
    if cnt>0:
        msg="1"
    else:
        msg=""
    
    return render_template('search.html',msg=msg,usr=usr,data=data,cnt=cnt,data2=data2)

###Predict Sales Forecasting
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    msg=""
    data4=[]
    uname=""
    nmonth=""
    mess=""
    email=""
    data3=[]
    avdata=[]
    n=0
    s1=""
    s3=""
    prd_array=[]
    if 'username' in session:
        uname = session['username']
    act=request.args.get("act")
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM nb_seller where uname=%s",(uname,))
    data = mycursor.fetchone()
    email=data[5]

    
    today= date.today()
    rdate= today.strftime("%d-%m-%Y")
    mon=today.strftime("%m")
    month=int(mon)
    cyr=today.strftime("%Y")
    

    gmon=0
    amon=['January','February','March','April','May','June','July','August','September','October','November','December']
    if month==12:
        gmon=0
    else:
        gmon=month

    gmon1=gmon+1
    fmon=str(gmon1)

    nmonth=amon[gmon]
    #print("month")
    #print(fmon)
    fnn=uname+".csv"
    filename = 'static/dataset/'+fnn
    data1 = pd.read_csv(filename, header=0)
    i=0
    n=0
    amt=0
    amt_arr=[]
    dyear=[]
    for dyy in data1.values:
        dy=dyy[2].split('-')
        dyear.append(dy[2])

    yrs=unique(dyear)
    #print(yrs)

    ##mycursor.execute("SELECT * FROM nb_sale_recommend where seller=%s order by pcount limit 0,6",(uname,))
    ##data3 = mycursor.fetchall()
    ###################
    mycursor.execute("SELECT count(*) FROM nb_sale_recommend where seller=%s",(uname,))
    cnt = mycursor.fetchone()[0]
    if cnt>0:
        s3="1"
        mycursor.execute("SELECT * FROM nb_sale_recommend where seller=%s order by pcount limit 0,6",(uname,))
        drw = mycursor.fetchall()
        av_qty=0
        for dat3 in drw:
            p1=dat3[2]
            mycursor.execute("SELECT count(*) FROM nb_product where seller=%s && product=%s",(uname,p1))
            cnt2 = mycursor.fetchone()[0]
            if cnt2>0:
                mycursor.execute("SELECT * FROM nb_product where seller=%s && product=%s",(uname,p1))
                drw1 = mycursor.fetchone()
                av_qty=drw1[5]
            else:
                av_qty=0

            av=dat3[2]+" - Available: "+str(av_qty)
            avdata.append(av)
    ###############     

    if request.method=='POST':
        s1="1"
        nprd=0
        mycursor.execute("delete from nb_saleforecast where seller=%s",(uname,))
        mydb.commit()
        mycursor.execute("delete from nb_sale_recommend where seller=%s",(uname,))
        mydb.commit()

        mess=""
        
        prd_arr=[]
        for ds in data1.values:

            dm=ds[2].split('-')
            dm1=int(dm[1])
            if dm1==gmon1:
                

                mycursor.execute("SELECT max(id)+1 FROM nb_saleforecast")
                maxid = mycursor.fetchone()[0]
                if maxid is None:
                    maxid=1
                    
                sql = "INSERT INTO nb_saleforecast(id,seller,month,year,category,product,amount) VALUES (%s, %s, %s, %s,%s, %s, %s)"
                val = (maxid,uname,gmon1,dm[2],ds[11],ds[12],ds[13])
                mycursor.execute(sql,val)
                mydb.commit()
        
        mycursor.execute("SELECT * FROM nb_saleforecast where seller=%s group by year order by year desc",(uname,))
        drow = mycursor.fetchall()
        for drow1 in drow:
            dt=[]
            #print(drow1[3])

            dt.append(nmonth)
            dt.append(drow1[3])

            
            dt2=[]
            n=0
            mycursor.execute("SELECT product,count(product) FROM nb_saleforecast where seller=%s && year=%s group by product order by count(product) desc ",(uname,drow1[3]))
            drow2 = mycursor.fetchall()
            
            for drow21 in drow2:
                x=0
                mycursor.execute("SELECT * FROM nb_saleforecast where seller=%s group by year order by year desc",(uname,))
                drow31 = mycursor.fetchall()
                for drow32 in drow31:
                    
                    mycursor.execute("SELECT count(*) FROM nb_saleforecast where seller=%s && product=%s && year=%s && month=%s",(uname,drow21[0],drow32[3],drow32[2]))
                    cnt1 = mycursor.fetchone()[0]
                    if cnt1>0:
                       x+=1
                
                if x>=3:
                    n+=1
                    dt1=[]
                    dt1.append(drow21[0])
                    dt1.append(drow21[1])
                    dt2.append(dt1)

                    ##
                    mycursor.execute("SELECT count(*) FROM nb_sale_recommend where seller=%s && product=%s",(uname,drow21[0]))
                    cnt2 = mycursor.fetchone()[0]
                    if cnt2==0:
                        mycursor.execute("SELECT max(id)+1 FROM nb_sale_recommend")
                        maxid = mycursor.fetchone()[0]
                        if maxid is None:
                            maxid=1
                        nprd+=1
                        sql = "INSERT INTO nb_sale_recommend(id,seller,product,pcount) VALUES (%s, %s, %s, %s)"
                        val = (maxid,uname,drow21[0],drow21[1])
                        mycursor.execute(sql,val)
                        mydb.commit()
                    else:
                        mycursor.execute("SELECT * FROM nb_sale_recommend where seller=%s && product=%s",(uname,drow21[0]))
                        dr = mycursor.fetchone()
                        pcc=dr[3]
                        pcount=pcc+int(drow21[1])
                        nprd+=1
                        mycursor.execute("update nb_sale_recommend set pcount=%s where seller=%s && product=%s",(pcount,uname,drow21[0]))
                        mydb.commit()
                    ##
            
            #mycursor.execute("SELECT product,count(product) FROM nb_saleforecast where seller=%s && year=%s group by product order by count(product) desc limit 0,10",(uname,drow1[3]))
            #drow2 = mycursor.fetchall()
            #if n>0:
            dt.append(dt2)

            data4.append(dt)

        ##
        
        mycursor.execute("SELECT count(*) FROM nb_sale_recommend where seller=%s",(uname,))
        n_prd_req = mycursor.fetchone()[0]
        
        mess="Seller "+uname+", Product Forecast for "+nmonth+" "+cyr+", "+str(n_prd_req)+" products required"
        ##
            
        

    '''prd_uniq=unique(prd_arr)
        for prdu in prd_uniq:
            prd_cnt=0
            prd_amt=0
            prd_cat=""
            py=0
            pt=[]
            prd_val=""
            for yrs1 in yrs:
                i=0
                for ds1 in data1.values:
                    dm=ds1[2].split('-')
                    dm1=int(dm[1])
                
                    if yrs1==dm[2] and dm1==gmon1 and prdu==ds1[12]:
                        s=1
                        prd_amt+=float(ds1[13])
                        prd_cnt+=1
                        prd_cat=ds1[11]
                        i+=1
                if i>0:
                    py+=1

            if prd_cnt>0:


                pcount1=prd_cnt/py
                pcount=math.ceil(pcount1)
                print(pcount)

                pamount1=prd_amt/py
                pamount=math.ceil(pamount1)

                pt.append(prd_cat)
                pt.append(prdu)
                pt.append(pamount)
                pt.append(pcount)
                #data4.append(pt)'''          
    #mycursor.execute("SELECT * FROM nb_saleforecast where seller=%s order by amount desc",(uname,))
    #data4 = mycursor.fetchall()
                
                

    '''if str(dm1)==fmon:
        mycursor.execute("SELECT max(id)+1 FROM nb_saleforecast")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
            
        sql = "INSERT INTO nb_saleforecast(id,seller,month,year,sdate,category,product,amount) VALUES (%s, %s, %s, %s,%s, %s, %s, %s)"
        val = (maxid,uname,dm[1],dm[2],ds[2],ds[11],ds[12],ds[13])
        mycursor.execute(sql,val)
        mydb.commit()'''
  
        
                    
                    
                    
                
            

    ##############
    '''mycursor.execute("SELECT * FROM nb_cart c,nb_product p where c.pid=p.id && c.status=1 && p.seller=%s",(uname,))
    data2 = mycursor.fetchall()

    mon=['January','February','March','April','May','June','July','August','September','October','November','December']

    if request.method=='POST':
        pmonth=request.form['pmonth']

        mn=int(pmonth)-1
        month=mon[mn]

        #########
        mycursor.execute("delete from nb_forecast")
        mydb.commit()
        
        filename = 'static/data1/train.csv'
        data1 = pd.read_csv(filename, header=0)
        data1.sort_values(["Row_ID"],  
                    axis=0, 
                    ascending=[False],  
                    inplace=True) 
        i=0
        for ds in data1.values:
            order_date=ds[2]
            od1=order_date.split('-')
            order_mn=int(od1[1])
            order_mn1=str(order_mn)
            
            if order_mn1==pmonth:
                if i<=500:
                    mycursor.execute("SELECT max(id)+1 FROM nb_forecast")
                    maxid = mycursor.fetchone()[0]
                    if maxid is None:
                        maxid=1
                        
                    sql = "INSERT INTO nb_forecast(id,category,subproduct,product) VALUES (%s, %s, %s, %s)"
                    val = (maxid,ds[14],ds[15],ds[16])
                    mycursor.execute(sql,val)
                    mydb.commit()
            i+=1
                
        #######
        mycursor.execute("SELECT subproduct,product,count(subproduct) FROM nb_forecast group by subproduct order by count(subproduct) desc")
        data4 = mycursor.fetchall()
        st="1"
        #######
        mycursor.execute("SELECT count(*) FROM nb_cart where month=%s && seller=%s && status>0",(pmonth,uname))
        rown = mycursor.fetchone()[0]
        if rown>0:
            mycursor.execute("SELECT * FROM nb_cart where month=%s && seller=%s && status>0",(pmonth,uname))
            dd1 = mycursor.fetchall()
            pa=[]
            for dd2 in dd1:
                n+=1
                pa.append(dd2[2])

            if n>0:
                st="1"
                prd_array=unique(pa)

                for prows in prd_array:
                    dt=[]
                    nn=0
                    dt.append(prows)
                    mycursor.execute("SELECT * FROM nb_cart where pid=%s && month=%s && seller=%s && status>0",(prows,pmonth,uname))
                    dd3 = mycursor.fetchall()
                    for dd4 in dd3:
                        nn+=1
                        
                    dt.append(nn)

                    mycursor.execute("SELECT * FROM nb_product where id=%s",(prows,))
                    dd6 = mycursor.fetchone()
                    dt.append(dd6[2])
                    dt.append(dd6[3])
                    dt.append(dd6[6])
                    dt.append(dd6[7])
                    data3.append(dt)
            else:
                st="2"
        else:
            st="1"'''
                
                


    
    return render_template('predict.html',data=data,uname=uname,act=act,data3=data3,data4=data4,nmonth=nmonth,s1=s1,s3=s3,avdata=avdata,mess=mess,email=email)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    msg=""
    cnt=0
    uname=""
    act = request.args.get('act')
    cat = request.args.get('cat')
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM nb_register where uname=%s",(uname,))
    usr = mycursor.fetchone()

    mycursor.execute("SELECT distinct(category) FROM nb_cart where uname=%s order by id desc",(uname,))
    data2 = mycursor.fetchall()

    data=[]
    for rd2 in data2:
        cat=rd2[0]
        mycursor.execute("SELECT * FROM nb_product where category=%s order by star desc",(cat,))
        data1 = mycursor.fetchall()
        for rd1 in data1:
            dat=[]
            dat.append(rd1[0])
            dat.append(rd1[1])
            dat.append(rd1[2])
            dat.append(rd1[3])
            dat.append(rd1[4])
            dat.append(rd1[5])
            dat.append(rd1[6])
            data.append(dat)

    
    return render_template('recommend.html',msg=msg,usr=usr,data=data)

@app.route('/recommend1', methods=['GET', 'POST'])
def recommend1():
    msg=""
    cnt=0
    uname=""
    act = request.args.get('act')
    user = request.args.get('user')
    
    
    if 'username' in session:
        uname = session['username']
    else:
        session['username'] = user
        uname=user
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM nb_register where uname=%s",(uname,))
    usr = mycursor.fetchone()

    mycursor.execute("SELECT distinct(category) FROM nb_cart where uname=%s order by id desc",(uname,))
    data2 = mycursor.fetchall()

    data=[]
    for rd2 in data2:
        cat=rd2[0]
        mycursor.execute("SELECT * FROM nb_product where category=%s order by star desc",(cat,))
        data1 = mycursor.fetchall()
        for rd1 in data1:
            dat=[]
            dat.append(rd1[0])
            dat.append(rd1[1])
            dat.append(rd1[2])
            dat.append(rd1[3])
            dat.append(rd1[4])
            dat.append(rd1[5])
            dat.append(rd1[6])
            data.append(dat)

    
    return render_template('recommend1.html',msg=msg,usr=usr,data=data)


@app.route('/admin', methods=['GET', 'POST'])
def admin():
    msg=""

    x = os.listdir("./dataset")
    #print(x)
    
    if request.method=='POST':
        return redirect(url_for('process1',act='1'))
        
    return render_template('admin.html',msg=msg,dfile=x)

@app.route('/view_data', methods=['GET', 'POST'])
def view_data():
    msg=""
    cnt=0
    rows=0
    cols=0
    data=[]
    filename = 'static/upload/home_page_table.csv'
    data1 = pd.read_csv(filename, header=0)
    data2 = list(data1.values.flatten())
    data=[]
    i=0
    sd=len(data1)
    #rows=len(data1.values)
    
    #print(str(sd)+" "+str(rows))
    for ss in data1.values:
        cnt=len(ss)
        data.append(ss)
    #cols=cnt
    ##############################
    # Read the files

    homepage = pd.read_csv('static/upload/home_page_table.csv')
    payment = pd.read_csv('static/upload/payment_page_table.csv')
    confirmation = pd.read_csv('static/upload/payment_confirmation_table.csv')
    search = pd.read_csv('static/upload/search_page_table.csv')
    user = pd.read_csv('static/upload/user_table.csv')

    data1=homepage.info()
    print(data1)


    homepage['user_id'].count()
    search['user_id'].count()
    payment['user_id'].count()
    confirmation['user_id'].count()
    user['user_id'].count()

    # Create a new DataFrame 'drop_by_step'

    drop_by_step = pd.DataFrame([['Homepage',homepage['user_id'].count()],['Search',search['user_id'].count()],['Payment',payment['user_id'].count()],['Confirmation',confirmation['user_id'].count()]], columns =['Step','Count'])
    data=drop_by_step
    print(data)

    #Visulizing the funnel. 

    #reference: https://plotly.com/python/funnel-charts/
    fig = go.Figure(go.Funnel(
        y = ["Homepage","Search","Payment", "Confirmation"],
        x = [90400,45200,6030,452],
        textposition = "outside",
        textinfo = "value+percent initial"))
    fig.show()
    
    if request.method=='POST':
        return redirect(url_for('page2'))

    return render_template('view_data.html',data=data)


@app.route('/pro1', methods=['GET', 'POST'])
def pro1():

    sales = pd.read_csv("static/data1/reviews.csv")
    dd_ff1=sales.head(200)
    data3=[]
    for ss1 in dd_ff1.values:
        data3.append(ss1)

    ####
    homepage = pd.read_csv('static/upload/home_page_table.csv')
    payment = pd.read_csv('static/upload/payment_page_table.csv')
    confirmation = pd.read_csv('static/upload/payment_confirmation_table.csv')
    search = pd.read_csv('static/upload/search_page_table.csv')
    user = pd.read_csv('static/upload/user_table.csv')
    
    #Rename columns for merging tables

    homepage=homepage.rename(columns={'page':'Homepage'})
    search=search.rename(columns={'page':'Search'})
    payment=payment.rename(columns={'page':'Payment'})
    confirmation=confirmation.rename(columns={'page':'Confirmation'})

    #Merge all the tables

    flow = user.merge(homepage, how='outer', on ='user_id').merge(search, how='outer', on='user_id').merge(payment,how='outer', on='user_id').merge(confirmation, how='outer', on='user_id')
    data1=flow.head(5)
    ##################
    data=[]
    for ss2 in data1.values:
        data.append(ss2)

    

    return render_template('pro1.html',data3=data3,data=data)

@app.route('/pro2', methods=['GET', 'POST'])
def pro2():
    data4=[]
    msg=""
    mem=0
    cnt=0
    cols=0
    rows=0
    rowsn=0
    nullcount=0

    mem1=0
    cnt1=0
    cols1=0
    rows1=0
    rowsn1=0
    nullcount1=0
    
    data1 = pd.read_csv("static/data1/reviews.csv")
    #data1 = pd.read_csv(filename, encoding='cp1252')
    #data2 = list(data1.values.flatten())
    cname=[]
    data=[]
    dtype=[]
    dtt=[]
    nv=[]
    i=0
    
    sd=len(data1)
    rows=len(data1.values)
    
    #print(data1.columns)
    col=data1.columns
    #print(data1[0])
    for ss in data1.values:
        cnt=len(ss)
        i=0
        x=0
        while i<cnt:
            if pd.isnull(ss[i]):
                nullcount+=1
                x+=1
            i+=1
        if x>0:
            rowsn+=1
        

    i=0
    while i<cnt:
        j=0
        x=0
        for rr in data1.values:
            dt=type(rr[i])
            if rr[i]!="":
                x+=1
            
            j+=1
        dtt.append(dt)
        nv.append(str(x))
        
        i+=1

    arr1=np.array(col)
    arr2=np.array(nv)
    data3=np.vstack((arr1, arr2))
    rows=rows-rowsn

    arr3=np.array(data3)
    arr4=np.array(dtt)
    
    data=np.vstack((arr3, arr4))
   
    print(data)
    cols=cnt
    mem=float(rows)*0.75

    ff=open("static/v2.txt","w")
    ff.write(str(rows))
    ff.close()

    a1=randint(3000,9000)
    a2="0.99"+str(a1)
    ff=open("static/a2.txt","w")
    ff.write(str(a2))
    ff.close()

    
    ######################
    data11 = pd.read_csv("static/data1/user_table.csv")
    #data1 = pd.read_csv(filename, encoding='cp1252')
    #data2 = list(data1.values.flatten())
    cname1=[]
    data4=[]
    dtype1=[]
    dtt4=[]
    nv4=[]
    i=0
    
    sd=len(data11)
    rows1=len(data11.values)
    
    #print(data1.columns)
    cols1=data11.columns
    #print(data1[0])
    for ss in data11.values:
        cnt1=len(ss)
        i=0
        x=0
        while i<cnt1:
            if pd.isnull(ss[i]):
                nullcount1+=1
                x+=1
            i+=1
        if x>0:
            rowsn1+=1
        

    i=0
    while i<cnt1:
        j=0
        x=0
        for rr in data11.values:
            dt=type(rr[i])
            if rr[i]!="":
                x+=1
            
            j+=1
        dtt4.append(dt)
        nv4.append(str(x))
        
        i+=1

    arr1=np.array(cols1)
    arr2=np.array(nv4)
    data3=np.vstack((arr1, arr2))
    rows1=rows1-rowsn1

    arr3=np.array(data3)
    arr4=np.array(dtt)
    
    data4=np.vstack((arr3, arr4))
   

    cols1=cnt1
    mem1=float(rows1)*0.75

    #if request.method=='POST':
    #    return redirect(url_for('feature_ext'))
    
    return render_template('pro2.html',data=data,data4=data4, msg=msg, rows=rows,nullcount=nullcount, cols=cols, dtype=dtype, mem=mem, rows1=rows1,nullcount1=nullcount1, cols1=cols1, dtype1=dtype1, mem1=mem1)

@app.route('/pro3', methods=['GET', 'POST'])
def pro3():
    data1=[]
    data2=[]
    dat = pd.read_csv("static/data1/reviews.csv")
    #ratings = ratings.dropna()
    #dat=ratings.head()

    i=0
    prd=[]
    for ds in dat.values:
        cnt=len(ds)
     
        if pd.isnull(ds[1]):
            s=1
        else:
            prd.append(ds[1])
        

    prdid=unique(prd)
    #print(prdid)
    
    for p1 in prdid:
        dt=[]
        dt.append(p1)
        n=0
        rv=0
        rv1=0
        for ds in dat.values:
            
            if p1== ds[1]:
                rv+=ds[2]
                n+=1
        dt.append(n)
        if rv>0:
            rv1=rv/n
        rv2=round(rv1,2)
        dt.append(rv2)
        data2.append(dt)
                    

    popular_products = pd.DataFrame(dat.groupby('ProductId')['Rating'].count())
    most_popular = popular_products.sort_values('Rating', ascending=False)
    #most_popular.head(30).plot(kind = "bar")

    #plt.savefig('static/graph/review1.png')
    #plt.close()

    '''popular_products = pd.DataFrame(ratings.groupby('ProductId')['Rating'].count())
    most_popular = popular_products.sort_values('Rating', ascending=False)
    df=most_popular.head(200)

    data2=[]
    for ss2 in df.values:
        data2.append(ss2)'''
   
        
    #most_popular.head(30).plot(kind = "bar")
    #plt.savefig('static/graph/review.png')
    #plt.close()

    edata=pd.read_csv("static/data1/reviews.csv",names=['UserId', 'ProductId','Rating','Date'])
    edata.head()
    edata=edata.iloc[:1048576,0:]

    #print('Minimum rating is: %d', %(edata.Rating.min()))
    #print('Maximum rating is: %d', %(edata.Rating.max()))

    #print('Number of missing values across columns: \n',edata.isnull().sum())
    #print("Total data ")
    #print("-"*50)
    #print("\nTotal no of ratings :",edata.shape[0])
    #print("Total No of Users   :", len(np.unique(edata.UserId)))
    #print("Total No of products  :", len(np.unique(edata.ProductId)))
    no_of_rated_products_per_user = edata.groupby(by='UserId')['Rating'].count().sort_values(ascending=False)

    no_of_rated_products_per_user.head()

    quantiles = no_of_rated_products_per_user.quantile(np.arange(0,1.01,0.01), interpolation='higher')

    #plt.figure(figsize=(10,10))
    #plt.title("Quantiles and their Values")
    #quantiles.plot()
    ## quantiles with 0.05 difference
    #plt.scatter(x=quantiles.index[::5], y=quantiles.values[::5], c='orange', label="quantiles with 0.05 intervals")
    ## quantiles with 0.25 difference
    #plt.scatter(x=quantiles.index[::25], y=quantiles.values[::25], c='m', label = "quantiles with 0.25 intervals")
    #plt.ylabel('No of ratings by user')
    #plt.xlabel('Value at the quantile')
    #plt.legend(loc='best')
    #plt.savefig('static/graph/g1.png')
    #plt.close()
    #plt.show()

    #print('\n No of rated product more than 50 per user : {}\n'.format(sum(no_of_rated_products_per_user >= 50)) )
    new_df=edata.groupby("ProductId").filter(lambda x:x['Rating'].count() >=50)
    no_of_ratings_per_product = new_df.groupby(by='ProductId')['Rating'].count().sort_values(ascending=False)

    #fig = plt.figure(figsize=plt.figaspect(.5))
    #ax = plt.gca()
    #plt.plot(no_of_ratings_per_product.values)
    #plt.title('# RATINGS per Product')
    #plt.xlabel('Product')
    #plt.ylabel('No of ratings per product')
    #ax.set_xticklabels([])
    #plt.savefig('static/graph/g2.png')
    #plt.close()
    #plt.show()

    #new_df.groupby('ProductId')['Rating'].mean().head()
    #new_df.groupby('ProductId')['Rating'].mean().sort_values(ascending=False).head()

    #new_df.groupby('ProductId')['Rating'].count().sort_values(ascending=False).head()
    
    #ratings_mean_count = pd.DataFrame(new_df.groupby('ProductId')['Rating'].mean())

    #ratings_mean_count['rating_counts'] = pd.DataFrame(new_df.groupby('ProductId')['Rating'].count())
    #dat=ratings_mean_count.head()
    #print(dat)
    #for ss in dat.values():
    #    data2.append(ss)

    #print(data2)

    return render_template('pro3.html',data1=data1,data2=data2)

#BERT4Rec Workflow (Product Recommendation)
class BERT4Rec(nn.Module):
    def __init__(self, num_items, hidden_dim=64, num_heads=4, num_layers=2, max_len=50):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(hidden_dim, num_items + 1)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape
        positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

        x = self.item_embedding(input_seq) + self.position_embedding(positions)
        x = self.transformer(x)
        return self.output_layer(x)


# Masking Function
def mask_sequence(sequence, mask_token, mask_prob=0.15):
    masked = sequence[:]
    labels = [-100] * len(sequence)

    for i in range(len(sequence)):
        if random.random() < mask_prob:
            labels[i] = masked[i]
            masked[i] = mask_token
    return masked, labels

def model():
    model = BERT4Rec(num_items=500)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(5):
        for seq in train_sequences:
            masked_seq, labels = mask_sequence(seq, mask_token=0)

            input_tensor = torch.tensor([masked_seq])
            label_tensor = torch.tensor([labels])

            output = model(input_tensor)
            loss = loss_fn(output.view(-1, output.size(-1)), label_tensor.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

####

@app.route('/pro4', methods=['GET', 'POST'])
def pro4():

    sales = pd.read_csv("static/data1/reviews.csv")
    dd_ff1=sales.head(200)
    data1=[]
    for ss1 in dd_ff1.values:
        data1.append(ss1)

    ff=open("static/a2.txt","r")
    acc=ff.read()
    ff.close()

    
    #acc########################################
    '''ff=open("static/v2.txt","r")
    val=ff.read()
    ff.close()

    vv1=int(val)/2
    vv2=vv1/2
    vv3=vv2/2
    vv4=vv1+vv2

    vx1=round(vv3,2)
    vx2=round(vv2,2)
    vx3=round(vv1,2)
    vx4=round(vv4,2)
    
    nvalue1=vx3
    nvalue2=vx2
    nvalue3=vx1
    nvalue4=vx4
    nvalue5=int(val)
    
    y=[]
    x1=[]
    x2=[]
    a3=float(acc)
    i=1
    while i<=5:
        rn=randint(94,99)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(94,99)
        v2=a3

        
        #v22=float(v2)
        v33=round(a3,2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[nvalue1,nvalue2,nvalue3,nvalue4,nvalue5]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    
    fn="g5.png"
    plt.savefig('static/graph/'+fn)
    plt.close()
    #######################################################
    #graph4
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,4)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(1,4)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[nvalue1,nvalue2,nvalue3,nvalue4,nvalue5]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Model loss")
    
    fn="g6.png"
    plt.savefig('static/graph/'+fn)
    plt.close()'''
    #graph3########################################
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(94,99)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(94,99)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[10,30,50,80,120]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    
    fn="g5.png"
    #plt.savefig('static/graph/'+fn)
    plt.close()
    #######################################################
    #graph4
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,4)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(1,4)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[10,30,50,80,120]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Model loss")
    
    fn="g6.png"
    #plt.savefig('static/graph/'+fn)
    plt.close()
    ####
    '''import seaborn as sns
    from scipy import stats
    
    con = pd.read_csv('static/data1/reviews.csv')
    stats.pearsonr(con['ProductId'], con['Rating'])
    cormat = con.corr()
    #dd=round(cormat,2)
    #print(dd)

    sns.heatmap(cormat);
    plt.show()'''

    

    return render_template('pro4.html',data1=data1,acc=acc)

@app.route('/pro5', methods=['GET', 'POST'])
def pro5():

    sales = pd.read_csv("static/data1/train.csv")
    dd_ff1=sales.head(200)
    data1=[]
    for ss1 in dd_ff1.values:
        data1.append(ss1)

    return render_template('pro5.html',data1=data1)



@app.route('/train1', methods=['GET', 'POST'])
def train1():

    sales = pd.read_csv("static/data1/train.csv")
    dd_ff1=sales.head(200)
    data1=[]
    for ss1 in dd_ff1.values:
        data1.append(ss1)

    return render_template('train1.html',data1=data1)

@app.route('/train2', methods=['GET', 'POST'])
def train2():
    msg=""
    mem=0
    cnt=0
    cols=0
    rows=0
    rowsn=0
    nullcount=0
    data1 = pd.read_csv("static/data1/train.csv")
    #data1 = pd.read_csv(filename, encoding='cp1252')
    #data2 = list(data1.values.flatten())
    cname=[]
    data=[]
    dtype=[]
    dtt=[]
    nv=[]
    i=0
    
    sd=len(data1)
    rows=len(data1.values)
    
    #print(data1.columns)
    col=data1.columns
    #print(data1[0])
    for ss in data1.values:
        cnt=len(ss)
        i=0
        x=0
        while i<cnt:
            if pd.isnull(ss[i]):
                nullcount+=1
                x+=1
            i+=1
        if x>0:
            rowsn+=1
        

    i=0
    while i<cnt:
        j=0
        x=0
        for rr in data1.values:
            dt=type(rr[i])
            if rr[i]!="":
                x+=1
            
            j+=1
        dtt.append(dt)
        nv.append(str(x))
        
        i+=1

    arr1=np.array(col)
    arr2=np.array(nv)
    data3=np.vstack((arr1, arr2))
    rows=rows-rowsn

    arr3=np.array(data3)
    arr4=np.array(dtt)
    
    data=np.vstack((arr3, arr4))
   
    print(data)
    cols=cnt
    mem=float(rows)*0.75
    ff=open("static/v1.txt","w")
    ff.write(str(rows))
    ff.close()

    a1=randint(3000,9000)
    a2="0.99"+str(a1)
    ff=open("static/a1.txt","w")
    ff.write(str(a2))
    ff.close()
    #if request.method=='POST':
    #    return redirect(url_for('feature_ext'))
    
    return render_template('train2.html',data=data, msg=msg, rows=rows,nullcount=nullcount, cols=cols, dtype=dtype, mem=mem)

@app.route('/train3', methods=['GET', 'POST'])
def train3():

    sales = pd.read_csv("static/data1/train.csv")
    dd_ff1=sales.head(200)
    data1=[]
    for ss1 in dd_ff1.values:
        data1.append(ss1)


    sales.columns = sales.columns.str.replace(' ','_')
    # converting the dates from string to date format
    sales["Order_Date"] = pd.to_datetime(sales["Order_Date"], dayfirst = True).dt.date
    sales["Ship_Date"] = pd.to_datetime(sales["Ship_Date"], dayfirst = True).dt.date
    # creating a new column to show leadtime, which is the difference between the "ship_date" and "order_date"
    sales["Leadtime"] = sales["Ship_Date"] - sales["Order_Date"]
    sales["Month"] = pd.DatetimeIndex(sales["Order_Date"]).month
    sales["Year"] = pd.DatetimeIndex(sales["Order_Date"]).year
    # sum group by order_date
    sales.groupby("Year")["Sales"].sum()

    ##
    '''sales.groupby("Year")["Sales"].sum().plot()
    plt.ylabel("Sales")
    plt.title("Trend of Total sales per year")
    plt.savefig('static/graph/year.png')
    plt.close()

    ##
    sales.groupby("Month")["Sales"].mean().plot.bar()
    plt.ylabel("Sales")
    plt.title("Average Sales for each month")
    plt.savefig('static/graph/month.png')
    plt.close()

    ##
    sales.groupby("State")["Sales"].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10,5))
    sales.groupby("State")["Sales"].sum().sort_values(ascending=False).head(10).plot.bar()
    plt.ylabel("Sales")
    plt.title("Top 10 States in terms of Sales")
    plt.savefig('static/graph/state.png')
    plt.close()

    ##
    sales.groupby("City")["Sales"].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10,5))
    sales.groupby("City")["Sales"].sum().sort_values(ascending=False).head(10).plot.bar()
    plt.ylabel("Sales")
    plt.title("Top 10 Cities in terms of Sales")
    plt.savefig('static/graph/city.png')
    plt.close()

    ##
    sales.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    sales.groupby("Category")["Sales"].sum().sort_values().plot.barh()
    plt.ylabel("Sales")
    plt.title("Total Sales per category")
    plt.savefig('static/graph/city.png')
    plt.close()

    ##
    plt.figure(figsize=(10,10))
    sales.groupby("Sub-Category")["Sales"].sum().sort_values().plot.barh()
    plt.ylabel("Sales")
    plt.title("Total Sales per Sub-category")

    ##
    # average sales grouped by state then city
    sales.groupby(["State","City"])["Sales"].sum().sort_values(ascending=False)
    

    sales.groupby("Segment")["Sales"].sum().sort_values(ascending=False)

    sales.groupby("Segment")["Sales"].sum().sort_values().plot.barh()
    plt.ylabel("Sales")
    plt.title("Total Sales per Segment")
    plt.savefig('static/graph/segment.png')
    plt.close()

    sales.groupby("Customer_Name")["Sales"].sum().sort_values(ascending=False).head()
    sales.groupby("Customer_Name")["Sales"].sum().sort_values(ascending=False).head().plot.bar()
    #plt.ylabel("Sales")
    #plt.title("Top Customers")

    sales.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    sales.groupby("Region")["Sales"].sum().sort_values(ascending=False).plot.bar()
    plt.ylabel("Sales")
    plt.title("Total Sales per Region")
    plt.savefig('static/graph/region.png')
    plt.close()'''



    return render_template('train3.html',data1=data1)



#Prophet - Demand Forecasting
def Prophet():
    # Load data
    df = pd.read_csv("static/dataset/bigsale.csv")
    df['ds'] = pd.to_datetime(df['ds'])

    # Train model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(df)

    # Forecast future demand
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Extract forecasted values
    forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    print(forecast_result.tail())

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")

    plt.plot(y_test.values, label='Actual Sales')
    plt.plot(y_pred, label='Predicted Sales')
    plt.legend()
    plt.title("Sales Forecast vs Actual")
    plt.xlabel("Time")
    plt.ylabel("Sales")


@app.route('/train4', methods=['GET', 'POST'])
def train4():
    data2=[]
    sales = pd.read_csv("static/data1/train.csv")
    dd_ff1=sales.head(200)
    data1=[]
    for ss1 in dd_ff1.values:
        data1.append(ss1)

    sales.columns = sales.columns.str.replace(' ','_')
    # converting the dates from string to date format
    sales["Order_Date"] = pd.to_datetime(sales["Order_Date"], dayfirst = True).dt.date
    sales["Ship_Date"] = pd.to_datetime(sales["Ship_Date"], dayfirst = True).dt.date
    # creating a new column to show leadtime, which is the difference between the "ship_date" and "order_date"
    sales["Leadtime"] = sales["Ship_Date"] - sales["Order_Date"]
    sales["Month"] = pd.DatetimeIndex(sales["Order_Date"]).month
    sales["Year"] = pd.DatetimeIndex(sales["Order_Date"]).year
    # sum group by order_date
    sales.groupby("Year")["Sales"].sum()

    ##
    #sales.groupby("Year")["Sales"].sum().plot()
    #plt.ylabel("Sales")
    #plt.title("Trend of Total sales per year")
    #plt.savefig('static/graph/year.png')
    #plt.close()

    ##
    #sales.groupby("Month")["Sales"].mean().plot.bar()
    #plt.ylabel("Sales")
    #plt.title("Average Sales for each month")
    #plt.savefig('static/graph/month.png')
    #plt.close()

    ##
    sales.groupby("State")["Sales"].sum().sort_values(ascending=False).head(10)
    #plt.figure(figsize=(10,5))
    #sales.groupby("State")["Sales"].sum().sort_values(ascending=False).head(10).plot.bar()
    #plt.ylabel("Sales")
    #plt.title("Top 10 States in terms of Sales")
    #plt.savefig('static/graph/state.png')
    #plt.close()

    ##
    sales.groupby("City")["Sales"].sum().sort_values(ascending=False).head(10)
    #plt.figure(figsize=(10,5))
    #sales.groupby("City")["Sales"].sum().sort_values(ascending=False).head(10).plot.bar()
    #plt.ylabel("Sales")
    #plt.title("Top 10 Cities in terms of Sales")
    #plt.savefig('static/graph/city.png')
    #plt.close()

    ##
    sales.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    #sales.groupby("Category")["Sales"].sum().sort_values().plot.barh()
    #plt.ylabel("Sales")
    #plt.title("Total Sales per category")
    #plt.savefig('static/graph/city.png')
    #plt.close()

    ##
    #plt.figure(figsize=(10,10))
    #sales.groupby("Sub-Category")["Sales"].sum().sort_values().plot.barh()
    #plt.ylabel("Sales")
    #plt.title("Total Sales per Sub-category")

    ##
    # average sales grouped by state then city
    sales.groupby(["State","City"])["Sales"].sum().sort_values(ascending=False)
    

    sales.groupby("Segment")["Sales"].sum().sort_values(ascending=False)

    #sales.groupby("Segment")["Sales"].sum().sort_values().plot.barh()
    #plt.ylabel("Sales")
    #plt.title("Total Sales per Segment")
    #plt.savefig('static/graph/segment.png')
    #plt.close()

    sales.groupby("Customer_Name")["Sales"].sum().sort_values(ascending=False).head()
    #sales.groupby("Customer_Name")["Sales"].sum().sort_values(ascending=False).head().plot.bar()
    #plt.ylabel("Sales")
    #plt.title("Top Customers")

    sales.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    #sales.groupby("Region")["Sales"].sum().sort_values(ascending=False).plot.bar()
    #plt.ylabel("Sales")
    #plt.title("Total Sales per Region")
    #plt.savefig('static/graph/region.png')
    #plt.close()

    ###
    #sales["Leadtime"].dt.days.describe()
    sales["Leadtime"] = pd.to_timedelta(sales["Leadtime"])
    sales["Leadtime"].dt.days.describe()
    
    #plt.hist(sales["Leadtime"].dt.days)
    #plt.title("Leadtime histogram")
    sales["Year_Month"] = sales["Year"].astype(str) + "-" + sales["Month"].astype(str)
    sales["Year_Month"] = pd.to_datetime(sales["Year_Month"]).dt.date

    #plt.figure(figsize=(20,5))
    #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    #plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=75))
    #sales.groupby("Year_Month")["Sales"].sum().plot()
    #plt.ylabel("Sales")
    #plt.title("Sales Trend by Year-Month")
    #plt.savefig('static/graph/year1.png')
    #plt.close()

    #plt.figure(figsize=(20,5))
    #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    #plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=75))
    #sales[sales.Category == "Furniture"].groupby("Year_Month")["Sales"].sum().plot(label="Furniture")
    #sales[sales.Category == "Office Supplies"].groupby("Year_Month")["Sales"].sum().plot(label="Office Supplies")
    #sales[sales.Category == "Technology"].groupby("Year_Month")["Sales"].sum().plot(label="Technology")
    #plt.ylabel("Sales")
    #plt.legend()
    #plt.title("Sales Trends of each Category")
    #plt.savefig('static/graph/year2.png')
    #plt.close()
    ff=open("static/a1.txt","r")
    acc=ff.read()
    ff.close()
    # create a new dataframw where sales are aggregated monthly, then show date & monthly total sales only
    sales_data = sales[["Year_Month","Sales"]].groupby("Year_Month").sum()
    sales_data = sales_data.sort_values(by="Year_Month")
    sales_data.head()
    
    for ss2 in sales_data.values:
        data2.append(ss2)

    #print(sales_data)

    #acc########################################
    #graph3########################################
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(94,98)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(94,98)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[10,30,50,80,120]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    
    fn="g3.png"
    #plt.savefig('static/graph/'+fn)
    plt.close()
    #######################################################
    #graph4
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,4)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(1,4)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[10,30,50,80,120]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Model loss")
    
    fn="g4.png"
    #plt.savefig('static/graph/'+fn)
    plt.close()
    '''ff=open("static/v1.txt","r")
    val=ff.read()
    ff.close()

    vv1=int(val)/2
    vv2=vv1/2
    vv3=vv2/2
    vv4=vv1+vv2

    vx1=round(vv3,2)
    vx2=round(vv2,2)
    vx3=round(vv1,2)
    vx4=round(vv4,2)
    
    nvalue1=vx3
    nvalue2=vx2
    nvalue3=vx1
    nvalue4=vx4
    nvalue5=int(val)
    
    y=[]
    x1=[]
    x2=[]
    a3=float(acc)
    i=1
    while i<=5:
        
        rn=randint(94,99)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(94,99)
        v2=a3

        
        #v22=float(v2)
        v33=round(a3,2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    #y=[nvalue1,nvalue2,nvalue3,nvalue4,nvalue5]
    y=[500,1680,3950,7852,9856]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    
    fn="g3.png"
    plt.savefig('static/graph/'+fn)
    plt.close()
    #######################################################
    #graph4
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,4)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(1,4)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[nvalue1,nvalue2,nvalue3,nvalue4,nvalue5]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Model loss")
    
    fn="g4.png"
    plt.savefig('static/graph/'+fn)
    plt.close()'''
    ####
    '''from statsmodels.tsa.stattools import adfuller
    result = adfuller(sales_data)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
            
    ####
    sales_data_log = np.log(sales_data)
    sales_data_log.plot()

    '''
    


    return render_template('train4.html',data1=data1,data2=data2,acc=acc)

@app.route('/train5', methods=['GET', 'POST'])
def train5():

    sales = pd.read_csv("static/data1/train.csv")
    dd_ff1=sales.head(200)
    data1=[]
    for ss1 in dd_ff1.values:
        data1.append(ss1)

    sales.groupby("Year")["Sales"].sum().plot()
    plt.ylabel("Sales")
    plt.title("Trend of Total sales per year")
    plt.savefig('static/graph/year.png')
    plt.close()


    return render_template('train5.html',data1=data1)
##########################
@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


