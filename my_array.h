using namespace std;



template <typename T>
T my_abs(T x){
	if (x > 0)
		return x;
	return -x;
}

template <typename T>
T my_prod(std::vector<T> &x){
	int N, r(1);
	N = x.size();
	for (int i = 0; i <N; i++)
		r *= x[i];
	return r;
};

template <typename T>
T my_prod(std::vector<T> &x, int start, int end = -1){
	int N, r(1);
	N = x.size();
	if (end == -1)
		end = N;
	assert(start > 0 & end < N+1);
	for (int i = start; i <end; i++)
		r *= x[i];
	return r;
};

// here we are going to store matrices. We intend to keep them as 1d objects that will pretend to be 2D or 3D
class Array{
public:
	std::vector<double> data;
	int n_dim;
	std::vector<int> dimensions;

	Array();

	// non-default constructors:
	Array(int);
	Array(int, int);
	Array(int, int, int);
	Array(int, std::vector<double>);
	Array(int, int, std::vector<double>);
	Array(int, int, int, std::vector<double>);

	// general number of dimensions
	Array(std::vector<int>);
	Array(std::vector<int>, std::vector<double>);

	// getters
	double get_value(int);
	double get_value(int, int);
	double get_value(int, int, int);
	double get_value(std::vector<int>);

	// soft getters - return zero if indices fall outside, needed to do convolution
	double soft_get_value(int);
	double soft_get_value(int, int);
	double soft_get_value(int, int, int);

	// filling in with data goes here
	void insert_value(int, double);
	void insert_value(int, int, double);
	void insert_value(int, int, int, double);
	void insert_value(int, std::vector<double>);

	void insert_value(int, Array);

	void extend(int);

	// debugging and output
	void dump_to_textfile(std::string);
	void print_first_values(int, std::string);
	void clear();

	// operators
	Array& operator*=(const double& mult){
		for (int i = 0; i < data.size(); i++)
			data[i] *= mult;
		return *this;
	}
};

Array::Array(){
	n_dim = 0;
	dimensions = {};
	data = {};
};

Array::Array(int size){
	n_dim = 1;

	this->dimensions.push_back(size);
	for (int i = 0; i < size; i++){
		this->data.push_back(0);
	};
};

Array::Array(int sizex, int sizey){
	this->n_dim = 2;
	this->dimensions.push_back(sizex);
	this->dimensions.push_back(sizey);
	for (int i = 0; i < sizex * sizey; i++)
		this->data.push_back(0);
};

Array::Array(int lines, int sizex, int sizey){
	this->n_dim = 3;
	this->dimensions.push_back(lines);
	this->dimensions.push_back(sizex);
	this->dimensions.push_back(sizey);
	for (int i = 0; i < lines * sizex * sizey; i++)
		this->data.push_back(0);
};

Array::Array(std::vector<int> dim_vec){
	if (dim_vec.size() == 1){
		this->dimensions.push_back(dim_vec[0]);
	} else if (dim_vec.size() == 2){
		this->dimensions.push_back(dim_vec[0]);
		this->dimensions.push_back(dim_vec[1]);
	} else if (dim_vec.size() == 3){
		this->dimensions.push_back(dim_vec[0]);
		this->dimensions.push_back(dim_vec[1]);
		this->dimensions.push_back(dim_vec[2]);
	} else {
		throw std::invalid_argument("dimension vector too large, currently we accept only up to three dimensions");
	};
};

double Array::get_value(int i){
	assert(i >= 0 && i < this->dimensions[0]);
	return this->data[i];
};

double Array::get_value(int line, int column){
	assert(line >= 0 && line < this->dimensions[0] && column >= 0 && column < this->dimensions[1]);
	return this->data[ this->dimensions[1] * line + column ];
};

double Array::get_value(int nr, int line, int column){
	assert(nr >= 0 && nr < this->dimensions[0] && line >= 0 && line < this->dimensions[1] && column >= 0 && column < this->dimensions[2]);
	return this->data[ nr * this->dimensions[1] * this->dimensions[2] + this->dimensions[2] * line + column ];
}

double Array::soft_get_value(int i){
	if (i >= 0 && i < this->dimensions[0])
		return this->data[i];
	return 0;
}

double Array::soft_get_value(int line, int column){
	if (line >= 0 && line < this->dimensions[0] && column >= 0 && column < this->dimensions[1])
		return this->data[ this->dimensions[1] * line + column ];
	return 0;
}

double Array::soft_get_value(int nr, int line, int column){
	if (nr >= 0 && nr < this->dimensions[0] && line >= 0 && line < this->dimensions[1] && column >= 0 && column < this->dimensions[2])
		return this->data[ nr * this->dimensions[1] * this->dimensions[2] + this->dimensions[2] * line + column ];
	return 0;
}

void Array::insert_value(int i, double val){
	assert(i >= 0 && i < this->dimensions[0] && this->n_dim == 1);
	this->data[i] = val;
};

void Array::insert_value(int i, int j, double val){
	assert(this->n_dim == 2);
	this->data[ this->dimensions[1] * i + j ] = val;
};

void Array::insert_value(int i, int j, int k, double val){
	assert(this->n_dim == 3);
	this->data[i * this->dimensions[1] * this->dimensions[2] + this->dimensions[2] * j + k ] = val;
};

void Array::insert_value(int line, std::vector<double> values){
	assert(this->n_dim == 2 and values.size() == this->dimensions[1]);

	for (int j = 0; j < this->dimensions[1]; j++){
		this->data[ this->dimensions[1] * line + j ] = values[j];
	};
};

void Array::dump_to_textfile(std::string filename){

	ofstream im_file;
	im_file.open(filename);
	if (this->n_dim == 1){
		for (int i = 0; i < this->data.size(); i++){
			im_file << this->data[i] << " ";
		};
	};
	if (this->n_dim == 2) {
		for (int i = 0; i < this->dimensions[0]; i++){
			for (int j = 0; j < this->dimensions[1]; j++){
				im_file << this->data[ i * this->dimensions[1] + j ] << " ";
			};
			im_file << endl;
		};
	};
	im_file.close();
};

void Array::print_first_values(int k, std::string name){
	assert(this->n_dim == 2);
	cout << "printing " << name << endl;
	int m = std::min<int>(k, this->dimensions[0]);
	for (int i = 0; i < m; i++){
		for (int j = 0; j < this->dimensions[1]; j++){
			cout << get_value(i, j) << " ";
		};
		cout << endl;
	};
};

void Array::extend(int length){
	if (this->n_dim == 1){
		for (int i = 0; i < length; i++)
			this->data.push_back(0);
		this->dimensions[0] += length;
	} else if (this->n_dim > 1){
		int mult = my_prod(dimensions, 1, -1);
		for (int i = 0; i < length; i++){
			for (int j = 0; j < mult; j++)
				this->data.push_back(0);
		};
		this->dimensions[0] += length;
	};
};

void Array::clear(){
	for (int i = 0; i < this->data.size(); i++)
		this->data[i] = 0;
}