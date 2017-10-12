from urllib2 import urlopen, URLError, HTTPError
from urllib import urlencode
from pyquery import PyQuery as pq
import re, json, sys, os, string
reload(sys)
sys.setdefaultencoding('utf-8')

def get(c):
	c = c.replace('phys','physics') if 'appphys' not in c and 'geophys' not in c else c
	c = c.replace('mse','ms%26')
	e = pq(os.popen('curl -s http://explorecourses.stanford.edu/search?q=%s&academicYear=20172018' % c.replace('-','')).read())
	rs = e("#searchResults h2")
	p = c.split('-')
	p[0] = p[0].replace('%26','&')
	for r in rs:
		r = e(r)
		n = r.find(".courseNumber").text()
		if p[0] in n.lower() and p[1] in n.lower():
			t = r.find(".courseTitle").text()
			tp = t.split('(')
			if len(tp) > 1:
				ce = tp[1].strip().split(" ")[0]
				if ce == ce.upper():
					t = t.split('(')[0].strip().replace('\n',' ')
			else:
				t = t.strip().replace('\n',' ')
			ca = r.parent().find(".courseAttributes")
			u = 4
			for ud in ca:
				u = e(ud).text()
				if "Units:" in u:
					u = u[u.find('Units:'):].split(":")[1].strip().split(" ")[0].strip()
					u = u if '-' not in u else u.split('-')[1]
					u = int(u)
				return [t,u]
	return getold(c,2016)

def getold(c,yr):
	dc = c
	c = c.replace('phys','physics') if 'appphys' not in c and 'geophys' not in c  else c
	c = c.replace('mse','ms%26')
	e = pq(os.popen('curl -s \'http://explorecourses.stanford.edu/search?q=%s&view=catalog&page=0&filter-coursestatus-Active=on&collapse=&academicYear=%d%d\'' % (c.replace('-',''),yr,yr+1)).read())
	rs = e("#searchResults h2")
	p = c.split('-')
	p[0] = p[0].replace('%26','&')
	for r in rs:
		r = e(r)
		n = r.find(".courseNumber").text()
		if p[0] in n.lower() and p[1] in n.lower():
			t = r.find(".courseTitle").text()
			tp = t.split('(')
			if len(tp) > 1:
				ce = tp[1].strip().split(" ")[0]
				if ce == ce.upper():
					t = t.split('(')[0].strip().replace('\n',' ')
			else:
				t = t.strip().replace('\n',' ')
			ca = r.parent().find(".courseAttributes")
			u = 4
			for ud in ca:
				u = e(ud).text()
				if "Units:" in u:
					u = u[u.find('Units:'):].split(":")[1].strip().split(" ")[0].strip()
					u = u if '-' not in u else u.split('-')[1]
					u = int(u)
				return [t,u]
	return [p[0].upper() + ' ' + p[1].upper(),4] if yr == 2011 else getold(dc,yr-1)