% Make class for the super-pixels
% def __init__(self):
% 		self.label=None
% 		self.pixels=[]
% 		self.mean_intensity=0.0
% 		self.centroid=()
% 		self.type='na'
% self.mean_lab=None
% 		self.lab_hist=None
% 		self.real_lab=None
% 	def __repr__(self):
% 		return str(self.label)

% above code is copied from that python file , needed for our reference 

%Tasks to do
%1. everything lol

% first - search for graph class, superpixels, tuples, scribbles image as
% input, histogram in matlab

classdef SPNode
    properties
        label=NaN;
        pixels
        mean_intensity
        centroid
        type
        mean_lab
        lab_hist
        real_hist
    end
    methods        
        function obj = SPNode()
            obj.pixels = [];
            obj.mean_intensity= 0.0;
            obj.type = 'na';
            obj.centroid = [];
            %initialise all class variables
        end
    end   
end
