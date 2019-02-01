#!/bin/bash
#software requerido:
module load software/bioinformatics/duk/2011
module load software/bioinformatics/emboss/6.6.0
file=$1
threads=$2
database=datafinalCTXM.fa
workdir=/home/dhceballosl/Projects/CTXM/
echo "executing: $0 $file $threads with database: $database"
#Para filtrar el metagenoma de referencia con la base de datos Consolidada de CTXM
rm -f duk_output_tmp
for k in 17 19 21 25 30 35 40 45 50 55 60 65
do 
  mkdir duk_${k}_kmer_tmp
  duk -k ${k} -m duk_${k}_kmer_tmp/salida_duk.fastq ${database} ${file} >> duk_output_tmp
done

#to process output file from duk
#Para procesar el archivo de salida de duk
better_kmer=`awk -F "||" -v kmer_r="" -v kmer_p="" -v reads="0" -v pvalue="0" -v kmer_aux="" '{
  if($0 ~ /#Mer size:/){
    split($0,lista_mer,":");
    kmer_aux=lista_mer[2]+0;
  }
  if($0 ~ /# Total number of matched reads:/){
    split($0,lista_mer,":");
    if(reads <= (lista_mer[2]+0) ){
      kmer_r = kmer_aux+0;
      reads = lista_mer[2]+0
    }
  }
  if($0 ~ /#Avg number of Kmer for each read:/){
    split($0,lista_mer,":");
    if(pvalue <= (lista_mer[2]+0) ){
      kmer_p = kmer_aux+0;
      pvalue = lista_mer[2]+0
    } 
  }
}END{
  if(kmer_r == kmer_p){
    print kmer_r
  }else{
    print kmer_p
  }
}' duk_output_tmp`

echo "the better kmer for duk is: $better_kmer"
mv duk_${better_kmer}_kmer_tmp duk_source_dir
rm -rf duk_*_kmer_tmp

#Para convertir de fastq a fasta
seqret -sequence duk_source_dir/salida_duk.fastq -outseq duk_source_dir/salida_duk.fasta

#Formateo del archivo fasta para la NN
sed 's/ /_/g' duk_source_dir/salida_duk.fasta | sed 's/|/+/g' > sequences.fa.tmp
rm -f class_nn_ctxm.txt
for sec in `grep '>' sequences.fa.tmp | sed 's/>//g'`
do
      seqret -sequence sequences.fa.tmp:${sec} -outseq sec.fa.tmp > /dev/null 2>&1
      grupo=`grep '>' sec.fa.tmp | cut -d "+" -f2 | cut -d "-" -f2`
      sec2=`grep -v '>' sec.fa.tmp | sed ':a;N;$!ba;s/\n/ /g'`
      echo "${sec2},${grupo}" >> training_nn_ctxm.txt
      rm -f sec.fa.tmp
done
rm -f *.tmp
#ejecución de script en python con la NN implementada en TensorFlow.

