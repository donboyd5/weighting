#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 17:43:12 2021

@author: donboyd
"""

from collections import namedtuple

def dict_nt(d):
    # convert dict to named tuple
    return namedtuple('ntd', sorted(d))(**d)

