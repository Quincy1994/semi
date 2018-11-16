#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>
#include <vector>
using namespace std;
typedef float real;

const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;

#define MAX_STRING 100
#define SIGMOID_BOUND 6
#define NEG_SAMPLING_POWER 0.75
const int neg_table_size = 1e8;
const int sigmoid_table_size = 1000;

// ============ parameters ==============
int num_threads = 1;
int num_negative = 5;
int dim = 100;
real init_rho = 0.025, rho;
long long uw_total_samples = 1, uw_current_sample_count = 0;

// ============ Hash_table ==============
const int w_hash_table_size = 30000000;
const int u_hash_table_size = 30000000;
int *w_vertex_hash_table, *u_vertex_hash_table, *l_vertex_hash_table;
real *sigmoid_table;

// =============== word information =========
struct WordVertex {
    double uw_degree;
    double lw_degree;
    char *name;
};

struct UserVertex {
    double u_degree;
    char *name;
};

struct LabelVertex {
    double l_degree;
    char *name;
};

char word_file[MAX_STRING], user_file[MAX_STRING];
char w_emb_file[MAX_STRING], u_emb_file[MAX_STRING];
struct WordVertex *w_vertex;
struct UserVertex *u_vertex;
int word_num_vertices = 0, user_num_vertices = 0;
int word_max_num_vertices = 1000, user_max_num_vertices = 1000;
real *w_emb, *u_emb, *l_emb;

// ================= edge information =============
char user_word_file[MAX_STRING];
int *uw_edge_source_id, *uw_edge_target_id;
long long uw_num_edges = 0;
double *uw_edge_weight;


// ============ edge sampling =====================
long long *uw_edge_alias;
double *uw_edge_prob;

// ============= negative table ====================
int *word_uw_neg_table, *word_lw_neg_table;


unsigned int W_Hash(char *w_key)
{
    unsigned int seed = 131;
    unsigned int hash = 0;
    while (*w_key)
    {
        hash = hash * seed + (*w_key++);
    }
    return hash % w_hash_table_size;
}

unsigned int U_Hash(char *u_key)
{
    unsigned int seed = 131;
    unsigned int hash = 0;
    while (*u_key)
    {
        hash = hash * seed + (*u_key++);
    }
    return hash % u_hash_table_size;
}

void InitWordHashTable()
{
    w_vertex_hash_table = (int *)malloc(w_hash_table_size * sizeof(int));
    for (int k=0; k != w_hash_table_size; k++) w_vertex_hash_table[k] = -1;
}

void InitUserHashTable()
{
    u_vertex_hash_table = (int *)malloc(u_hash_table_size * sizeof(int));
    for (int k=0; k != u_hash_table_size; k++) u_vertex_hash_table[k] = -1;
}

void InsertWordHashTable(char *w_key, int value)
{
    int addr = W_Hash(w_key);
    while (w_vertex_hash_table[addr] != -1) addr = (addr + 1) % w_hash_table_size;
    w_vertex_hash_table[addr] = value;
}

void InsertUserHashTable(char *u_key, int value)
{
    int addr = U_Hash(u_key);
    while (u_vertex_hash_table[addr] != -1) addr = (addr + 1) % u_hash_table_size;
    u_vertex_hash_table[addr] = value;
}

int SearchWordHashTable(char *w_key)
{
    int addr = W_Hash(w_key);
    while (1)
    {
        if (w_vertex_hash_table[addr] == -1) return -1;
        if (!strcmp(w_key, w_vertex[w_vertex_hash_table[addr]].name)) return w_vertex_hash_table[addr];
        addr = (addr + 1) % w_hash_table_size;
    }
}

int SearchUserHashTable(char *u_key)
{
    int addr = U_Hash(u_key);
    while (1)
    {
        if (u_vertex_hash_table[addr] == -1) return -1;
        if (!strcmp(u_key, u_vertex[u_vertex_hash_table[addr]].name)) return u_vertex_hash_table[addr];
        addr = (addr + 1) % u_hash_table_size;
    }
}

int AddWordVertex(char *w_name)
{
    int length = strlen(w_name) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    w_vertex[word_num_vertices].name = (char *)calloc(length, sizeof(char));
    strncpy(w_vertex[word_num_vertices].name, w_name, length - 1);
    w_vertex[word_num_vertices].uw_degree = 0;
    w_vertex[word_num_vertices].lw_degree = 0;

    word_num_vertices ++;
    if (word_num_vertices + 2 >= word_max_num_vertices)
    {
        word_max_num_vertices += 1000;
        w_vertex = (struct WordVertex *)realloc(w_vertex, word_max_num_vertices * sizeof(struct WordVertex));
    }
    InsertWordHashTable(w_name, word_num_vertices - 1);
    return word_num_vertices -1;
}

// add a user vertex to the vertex set
int AddUserVertex(char *u_name)
{
    int length = strlen(u_name) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    u_vertex[user_num_vertices].name = (char *)calloc(length, sizeof(char));
    strncpy(u_vertex[user_num_vertices].name, u_name, length - 1);
    u_vertex[user_num_vertices].u_degree = 0;

    user_num_vertices ++;
    if (user_num_vertices + 2 >= user_max_num_vertices)
    {
        user_max_num_vertices += 1000;
        u_vertex = (struct UserVertex *)realloc(u_vertex, user_max_num_vertices * sizeof(struct UserVertex));
    }
    InsertUserHashTable(u_name, user_num_vertices - 1);
    return user_num_vertices - 1;
}

void LoadWordVertex()
{
    FILE *fin;
    char w_name[MAX_STRING], w_index[MAX_STRING], str[2 * MAX_STRING + 10000];
    int vid_w;
    int num_nodes = 0;

    fin = fopen(word_file, "rb");
    if (fin == NULL)
    {
        printf("error: word nodes file not found\n");
        exit(1);
    }
    while(fgets(str, sizeof(str), fin)) num_nodes++;
    fclose(fin);
    fin = fopen(word_file, "rb");
    for (int k = 0; k != num_nodes; k++)
    {
        fscanf(fin, "%s %s", w_name, w_index);
        if(k % 1000 == 0)
        {
            printf("Reading words: %.3lf%%%c", k / (double)(num_nodes + 1) * 100, 13);
            fflush(stdout);
        }
        vid_w = SearchWordHashTable(w_index);
        if (vid_w == -1) vid_w = AddWordVertex(w_index);
    }
    fclose(fin);
    printf("Number of word vertices: %d         \n", word_num_vertices);
}

void LoadUserVertex()
{
    FILE *fin;
    char u_name[MAX_STRING], u_index[MAX_STRING], str[2 * MAX_STRING + 10000];
    int vid_u;
    int num_nodes = 0;

    fin = fopen(user_file, "rb");
    if (fin == NULL)
    {
        printf("error: user nodes file not found\n");
        exit(1);
    }
    while(fgets(str, sizeof(str), fin)) num_nodes++;
    fclose(fin);
    fin = fopen(user_file, "rb");
    for (int k = 0; k != num_nodes; k++)
    {
        fscanf(fin, "%s %s", u_name, u_index);
        if(k % 1000 == 0)
        {
            printf("Reading users: %.3lf%%%c", k / (double)(num_nodes + 1) * 100, 13);
            fflush(stdout);
        }
        vid_u = SearchUserHashTable(u_index);
        if (vid_u == -1) vid_u = AddUserVertex(u_index);
    }
    fclose(fin);
    printf("Number of user vertices: %d         \n", user_num_vertices);
}

void LoadUWEdges()
{
    FILE *fin;
    char u_name[MAX_STRING], w_name[MAX_STRING], str[2 * MAX_STRING + 10000];
    int vid_u, vid_w;
    double weight;
    fin = fopen(user_word_file, "rb");
    if (fin == NULL)
    {
        printf("ERROR: user word file not found\n");
        exit(1);
    }
    uw_num_edges = 0;
    while(fgets(str, sizeof(str), fin)){
        uw_num_edges++;
    }
    fclose(fin);
    printf("Number of user word edges: %lld         \n", uw_num_edges++);

    uw_edge_source_id = (int *)malloc(uw_num_edges * sizeof(int));
    uw_edge_target_id = (int *)malloc(uw_num_edges * sizeof(int));
    uw_edge_weight = (double *)malloc(uw_num_edges * sizeof(double));
    if (uw_edge_source_id == NULL || uw_edge_target_id == NULL || uw_edge_weight == NULL){
        printf("Error: memory allocation failed!\n");
        exit(1);
    }
    fin = fopen(user_word_file, "rb");
    for(int k = 0; k != uw_num_edges; k++){
        fscanf(fin, "%s %s %lf", u_name, w_name, &weight);
        if (k % 10000 == 0){
            printf("Reading user word edges: %.3lf%%%c", k / (double)(uw_num_edges + 1) * 100, 13);
            fflush(stdout);
        }
        vid_u = SearchUserHashTable(u_name);
        uw_edge_source_id[k] = vid_u;
        vid_w = SearchWordHashTable(w_name);
        uw_edge_target_id[k] = vid_w;
        uw_edge_weight[k] = weight;
        u_vertex[vid_u].u_degree += weight;
        w_vertex[vid_w].uw_degree += weight;
    }
    fclose(fin);
    printf("User word edges have been loaded.\n");
}

void InitUWEdgeAliasTable()
{
    uw_edge_alias = (long long *)malloc(uw_num_edges * sizeof(long long));
    uw_edge_prob = (double *)malloc(uw_num_edges * sizeof(long long ));
    if (uw_edge_alias == NULL || uw_edge_prob == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }

    double *norm_prob = (double *)malloc(uw_num_edges * sizeof(double));
    long long *large_block = (long long*)malloc(uw_num_edges * sizeof(long long));
    long long *small_block = (long long*)malloc(uw_num_edges * sizeof(long long));
    if (norm_prob == NULL || large_block == NULL || small_block == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }

    double sum = 0;
    long long cur_small_block, cur_large_block;
    long long num_small_block = 0, num_large_block = 0;

    for(long long k=0; k != uw_num_edges; k++) sum += uw_edge_weight[k];
    for(long long k=0; k != uw_num_edges; k++) norm_prob[k] = uw_edge_weight[k] * uw_num_edges / sum;

    for(long long k = uw_num_edges - 1; k>=0; k--)
    {
        if(norm_prob[k] < 1)
            small_block[num_small_block++] = k;
        else
            large_block[num_large_block++] = k;
    }

    while(num_small_block && num_large_block)
    {
        cur_small_block = small_block[--num_small_block];
        cur_large_block = large_block[--num_large_block];
        uw_edge_prob[cur_small_block] = norm_prob[cur_small_block];
        uw_edge_alias[cur_small_block] = cur_large_block;
        norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
        if (norm_prob[cur_large_block] < 1)
            small_block[num_small_block++] = cur_large_block;
        else
            large_block[num_large_block++] = cur_large_block;
    }
    while (num_large_block) uw_edge_prob[large_block[--num_large_block]] = 1;
    while (num_small_block) uw_edge_prob[small_block[--num_small_block]] = 1;
    free(norm_prob);
    free(small_block);
    free(large_block);
}

/* Initialize the vertex embedding */
void InitWordVector()
{
    long long w, b;

    w = posix_memalign((void **)&w_emb, 128, (long long)word_num_vertices * dim * sizeof(real));
    if (w_emb == NULL) {printf("Error: memory allocation failed\n"); exit(1);}
    for(b = 0; b < dim; b++) for(w =0; w < word_num_vertices; w++)
        w_emb[w * dim + b] = (rand() / (real) RAND_MAX - 0.5) / dim;

}

void InitUserVector()
{
    long long u, b;
    u = posix_memalign((void **)&u_emb, 128, (long long)user_num_vertices * dim * sizeof(real));
    if (u_emb == NULL) {printf("Error: memory allocation failed\n"); exit(1);}
    for(b = 0; b < dim; b++) for(u = 0; u < user_num_vertices; u++)
        u_emb[u * dim + b] = (rand() / (real) RAND_MAX - 0.5 ) / dim;
}

/* Sample negative word samples according to vertex degrees */
void InitUWNegTable()
{
    double sum = 0, cur_sum = 0, por = 0;
    int vid = 0;
    word_uw_neg_table = (int *)malloc(neg_table_size * sizeof(int));
    for (int k = 0; k != word_num_vertices; k++) sum += pow(w_vertex[k].uw_degree, NEG_SAMPLING_POWER);
    for (int k = 0; k != neg_table_size; k++){
        if ((double)(k + 1)/ neg_table_size > por)
        {
            cur_sum += pow(w_vertex[vid].uw_degree, NEG_SAMPLING_POWER);
            por = cur_sum / sum;
            vid++;
        }
        word_uw_neg_table[k] = vid - 1;
    }
}

/* Fastly compute sigmoid function */
void InitSigmoidTable()
{
    real x;
    sigmoid_table = (real *)malloc((sigmoid_table_size + 1) * sizeof(real));
    for (int k = 0; k != sigmoid_table_size; k++){
        x = 2.0 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
        sigmoid_table[k] = 1.0 / (1.0 + exp(-x));
    }
}

real FastSigmoid(real x)
{
    if (x > SIGMOID_BOUND) return 1;
    else if (x < -SIGMOID_BOUND) return 0;
    int k = (x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2;
    return sigmoid_table[k];
}

void Update(real *vec_u, real *vec_v, real *vec_error, int label)
{
    real x = 0, g;
    for (int c = 0; c != dim; c++) x += vec_u[c] * vec_v[c];
    g = (label - FastSigmoid(x)) * rho;
    for (int c = 0; c != dim; c++) vec_error[c] += g * vec_v[c];
    for (int c = 0; c != dim; c++) vec_v[c] += g * vec_u[c];
}

long long SampleUWAnEdge(double rand_value1, double rand_value2)
{
    long long k = (long long)uw_num_edges * rand_value1;
    return rand_value2 < uw_edge_prob[k] ? k : uw_edge_alias[k];
}

/* Fastly generate a random integer */
int Rand(unsigned long long &seed)
{
    seed = seed * 25214903917 + 11;
    return (seed >> 16) % neg_table_size;
}

void *TrainUWLINEThread(void *id)
{
    long long u, w, lu, lw, target, label;
    long long count = 0, last_count = 0, curedge;
    unsigned long long seed = (long long)id;
    real *vec_error = (real *)calloc(dim, sizeof(real));

    while (1)
    {
        if (count > uw_total_samples / num_threads + 2) break;

        if (count - last_count > 10000)
        {
            uw_current_sample_count += count - last_count;
            last_count = count;
            printf("%cRho: %f Progress: %.3lf%%", 13, rho, (real)uw_current_sample_count / (real)(uw_total_samples + 1) * 100);
            fflush(stdout);
            rho = init_rho * (1 - uw_current_sample_count / (real)(uw_total_samples + 1));
            if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
        }

        curedge = SampleUWAnEdge(gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r));
        u = uw_edge_source_id[curedge];
        w = uw_edge_target_id[curedge];

        lu = u * dim;
        for (int c = 0; c != dim; c++) vec_error[c] = 0;

        // negative sampling
        for (int d = 0; d != num_negative + 1; d++){
            if (d == 0)
            {
                target = w;
                label = 1;
            }
            else{
                do{
                    target = word_uw_neg_table[Rand(seed)];
                }while(target == w);
                label = 0;
            }
            lw = target * dim;
            Update(&u_emb[lu], &w_emb[lw], vec_error, label);
        }
        for(int c=0; c != dim; c++) u_emb[c + lu] += vec_error[c];

        count++;
    }
    free(vec_error);
    pthread_exit(NULL);
}

void Output_word_emb()
{
    FILE *fo = fopen(w_emb_file, "wb");
    fprintf(fo, "%d %d\n", word_num_vertices, dim);
    for (int a =0; a < word_num_vertices; a++){
        fprintf(fo, "%s ", w_vertex[a].name);
        for(int b = 0; b < dim; b++) fprintf(fo, "%lf ", w_emb[a * dim + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);
}

void Output_user_emb()
{
    FILE *fo = fopen(u_emb_file, "wb");
    fprintf(fo, "%d %d\n", user_num_vertices, dim);
    for(int a = 0; a < user_num_vertices; a++){
        fprintf(fo, "%s ", u_vertex[a].name);
        for(int b =0; b < dim; b++) fprintf(fo, "%lf ", u_emb[a * dim + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);
}


void StepOneTrain()
{
    long a;
    pthread_t *pt = (pthread_t *) malloc(num_threads * sizeof(pthread_t));

    printf("--------------------------------\n");
    printf("UWSamples: %lldM\n", uw_total_samples / 1000000);
    printf("Negative: %d\n", num_negative);
    printf("Dimension: %d\n", dim);
    printf("Initial rho: %lf\n", init_rho);
    printf("--------------------------------\n");

    InitWordHashTable();
    InitUserHashTable();

    LoadWordVertex();
    LoadUserVertex();
    LoadUWEdges();
    InitUWEdgeAliasTable();
    InitWordVector();
    InitUserVector();
    InitUWNegTable();
    InitSigmoidTable();

    gsl_rng_env_setup();
    gsl_T = gsl_rng_rand48;
    gsl_r = gsl_rng_alloc(gsl_T);
    gsl_rng_set(gsl_r, 314159265);

    clock_t start = clock();
    printf("----------------------------\n");
    for(a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainUWLINEThread, (void *)a);
    for(a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    printf("\n");
    clock_t finish = clock();
    printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);

    Output_word_emb();
    Output_user_emb();
}

int ArgPos(char *str, int argc, char ** argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])){
        if (a == argc - 1){
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char ** argv) {
    int i;
    if (argc == 1){
        printf("StepOne: training user and word embedding in an unsupervised framework.");
        printf("Option:\n");
        printf("Parameters for training:\n");

        printf("\t-userVocab <file>\n");
        printf("\t\tUse user vocab to insert user nodes into the network\n");
        printf("\t-wordVocab <file>\n");
        printf("\t\tUse word vocab to insert word nodes into the network\n");
        printf("\t-uwEdges <file>\n");
        printf("\t\tUser user word edges to train the model\n");

        printf("\t-wordemb <file>\n");
        printf("\t\tUse <file> to save the learnt word embeddings\n");
        printf("\t-useremb <file>\n");
        printf("\t\tUse <file> to save the learnt user embeddings\n");

        printf("\t-dim <int>\n");
        printf("\t\tSet dimension of vertex embeddings; default is 100\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5\n");
        printf("\t-uwsamples <int>\n");
        printf("\t\tSet the number of uw training samples as <int>Million; default is 1\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 1)\n");
        printf("\t-rho <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025\n");
        return 0;
    }
    if ((i = ArgPos((char *)"-userVocab", argc, argv)) > 0) strcpy(user_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-wordVocab", argc, argv)) > 0) strcpy(word_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-uwEdges", argc, argv)) > 0) strcpy(user_word_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-wordemb", argc, argv)) > 0) strcpy(w_emb_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-useremb", argc, argv)) > 0) strcpy(u_emb_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-dim", argc, argv)) > 0) dim = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) num_negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-uwsamples", argc, argv)) > 0) uw_total_samples = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-rho", argc, argv)) > 0) init_rho = atof(argv[i + 1]);
    uw_total_samples *= 1000000;
    rho = init_rho;
    w_vertex = (struct WordVertex *)calloc(word_max_num_vertices, sizeof(struct WordVertex));
    u_vertex = (struct UserVertex *)calloc(user_max_num_vertices, sizeof(struct UserVertex));
    StepOneTrain();
    return 0;
}
















