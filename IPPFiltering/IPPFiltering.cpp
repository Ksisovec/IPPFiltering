#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <iostream>

#include <intrin.h>
#include "ippcore.h"
#include "ipps.h"

#include "RecordAndProcessing_filtering_IppFiltering.h"
/*
Ipp8u - unsigned char
ipp32f - float
Ipp64f - double
IppStatus - indicate status type, something like this:
- Indicates no error.
- Indicates an error when one of the specified pointers with exception of pBuffer is NULL.
- Indicates an error when no memory is allocated.
and i.e.
*/

/*
Вычисление коэфициентов фильтра
parametrs:
order - порядок фильтра
rLowFreq - отношение нижней граничащей частоты к частоте дискретизации
rHighFreq - отношение верхней граничащей частоты к частоте дискретизации
return:
pTaps - коэфициенты фильтра
*/
Ipp64f* firGenBandpass(int order, Ipp64f rLowFreq, Ipp64f rHighFreq)
{
	IppStatus status;
	Ipp8u* pBuffer;
	Ipp64f* pTaps_64f;
	int size;
	if ((status = ippsFIRGenGetBufferSize(order, &size)) != ippStsNoErr) {
		printf("\n-- Error %d, %s", status, ippGetStatusString(status));
		return NULL;
	}
	printf("\n pbufferSize = %d", size);
	pBuffer = ippsMalloc_8u(size);
	pTaps_64f = ippsMalloc_64f(order);
	Ipp64f* pTaps = ippsMalloc_64f(order);
	/*
	rLowFreq и rHighFreq - верзняя и нижняя частотные границы
	pTaps_64f - Указатель на массив в котором хранятся вычесленные значения (результат)
	order - колчиество элементов в массиве содержащих значения, должен быть больше или равен 5
	т.е. судя по всему размер pTaps_64f
	ippWinHamming/ippWinBlackman  - тип используемой оконнной функции
	ippTrue - нормализация (True - вкл)
	pBuffer - указатель на буфер для внутренних вычислений
	*/
	/* генерация коэфициенто фильтра 
	результат в pTaps_64f */
	if ((status = ippsFIRGenBandpass_64f(rLowFreq, rHighFreq, pTaps_64f, order, ippWinHamming, ippTrue, pBuffer)) != ippStsNoErr) {
		printf("\n-- Error %d, %s", status, ippGetStatusString(status));
		printf("\n rlowfreq = %f \n rhighfreq = %f \n order = %d \n", rLowFreq, rHighFreq, order);
		return NULL;
	}

	for (int i = 0; i < order; i++) {
		pTaps[i] = pTaps_64f[i];
	}
	ippsFree(pTaps_64f);
	return pTaps;
}
/*
Фильтрация (если есть 2 и более сдоступных потока)
parametrs:
src - указатель на вектор источника
dst - указатель на вектор результата
len - длинна источника
pSpec - указатель на внутреннюю структуру инструкций
pDlySrc - delay line
pDluDst - output delay line
pBuffer - указатель на рабочий буфер
NTHREADS - количество доступных потоков
bufSize - размер рабочего буфера
return:
dst - указатель на вектор результата
*/
void fir_omp(Ipp64f* src, Ipp64f* dst, int len, int order, IppsFIRSpec_64f* pSpec, Ipp64f* pDlySrc,
	Ipp64f* pDlyDst, Ipp8u* pBuffer, int NTHREADS, int bufSize)
{
	int  tlen, ttail;
	tlen = len / NTHREADS;		//количество данных обрабатываемых каждым потоком за раз
								/*остаток данных, что обработаються в последнем потокею. Т.е. последний поток обрабатывает tlen+ttail количество данных*/
	ttail = len % NTHREADS;
	/*код ниже выполняется в потоках, количество потоков от 1 до NTHREADS*/
#pragma omp parallel num_threads(NTHREADS) 
	{

		//printf("thread num = %5d\n", omp_get_thread_num);		//for debug
		int id = omp_get_thread_num();
		//printf("thread num = %5d\n", id);		//for debug
		Ipp64f* s = src + id*tlen;
		Ipp64f* d = dst + id*tlen;
		/*Размер данных отправляемых в каждый поток = весьМассив/количествоПотоков
		в последнем выполняемом потоке(последнем по id) к данным отправленным на фильтрацию добавляются остаточные данные
		т.е. остаток от весьМассив/количествоПотоков*/
		int len = tlen + ((id == NTHREADS - 1) ? ttail : 0);
		Ipp8u* b = pBuffer + id*bufSize;

		if (id == 0)
			ippsFIRSR_64f(s, d, len, pSpec, pDlySrc, NULL, b);
		else if (id == NTHREADS - 1)
			ippsFIRSR_64f(s, d, len, pSpec, s - (order - 1), pDlyDst, b);
		else
			ippsFIRSR_64f(s, d, len, pSpec, s - (order - 1), NULL, b);
	}
}

/*
parametrs:
order - поярдок фильтра 
data - исходные данные
len - количества исходных данных
rLowFreq - отношение нижней граничащей частоты к частоте дискретизации
rHighFreq - отношение нижней граничащей частоты к частоте дискретизации
return:
data - отфильтравынные данные
*/
float* filtering(int order, float* data, int len, Ipp64f rLowFreq, Ipp64f rHighFreq)
{
	IppStatus status;		//либо сигнализирует об ошибке, либо о том, что её нет
	Ipp64f* pSrc;			//указатель на вектор источника
	Ipp64f* pDst;			//указатель на вектор результата
							/*Указатель на внутреннюю структуру инструкций*/
	IppsFIRSpec_64f* pSpec;

	/*https://habr.com/company/intel/blog/276687/*/
	/**/
	Ipp64f*  pDlySrc = NULL;/*initialize delay line with "0"*/
							/**/
	Ipp64f*  pDlyDst = NULL;/*don't write  output delay line*/

	Ipp8u* pBuffer;			//указатель на рабочий буфер
	IppAlgType algType = ippAlgFFT; //алгоритм фильтрации (direct/FFT)

	int specSize;			//указатель на рамзмер внутренней структуры инструкций

							/*allocate memory for input and output vectors*/
	pSrc = ippsMalloc_64f(len);
	pDst = ippsMalloc_64f(len);

	/* копирование исходных данных в массив для фильтрации */
	for (int i = 0; i < len; i++) {
		pSrc[i] = data[i];
	}
	//memcpy(pSrc, data, len * sizeof(data[0]));

	/*генерация коэфициентов фильтра*/
	Ipp64f* pTaps = firGenBandpass(order, rLowFreq, rHighFreq);

	/*рамзмер рабочего буфера*/
	int bufSize;
	/*вычисляется рамер буфера для spec и buf*/
	if ((status = ippsFIRSRGetSize(order, ipp64f, &specSize, &bufSize)) != ippStsNoErr) {
		printf("\n-- Error %d, %s", status, ippGetStatusString(status));
		return NULL;
	}
	/*allocate memory for pSpec*/
	pSpec = (IppsFIRSpec_64f*)ippsMalloc_8u(specSize);

	/*определение максимально допустимого количества потоков*/
	omp_set_dynamic(1);			//включение динамического вычисления количества доступных потоков
	if (omp_get_dynamic() == 0)
		printf("Dynamic change thread num dissabled :(\n");
	//максимально допустимое число потоков для использования в следующей параллельной области
	int NTHREADS = omp_get_max_threads();
	//printf("NTHREADS = %d\n", NTHREADS);


	/*для N потоков bufSize должен быть увеличен в N раз*/
	/*выделение памяти в количестве bufSize*NTHREADS bytes*/
	pBuffer = ippsMalloc_8u(bufSize*NTHREADS);

	/*инициализация pSpec(внутренней структуры инструкций)*/
	if ((status = ippsFIRSRInit_64f(pTaps, order, algType, pSpec)) != ippStsNoErr) {
		printf("\n-- Error %d, %s", status, ippGetStatusString(status));
		return NULL;
	}
	/*apply FIR filter*/
	/*start measurement for sinle threaded*/
	if (NTHREADS == 1) {
		ippsFIRSR_64f(pSrc, pDst, len, pSpec, pDlySrc, pDlyDst, pBuffer);
	}
	else {
		/* потоковая фильтрация */
		fir_omp(pSrc, pDst, len, order, pSpec, pDlySrc, pDlyDst, pBuffer, NTHREADS, bufSize);
	}

	for (int i = 0; i < len; i++) {
		data[i] = pDst[i];
	}

	ippsFree(pSrc);
	ippsFree(pDst);
	ippsFree(pTaps);
	ippsFree(pSpec);
	ippsFree(pBuffer);

	return data;
}
/*
parametrs:
order - поярдок фильтра
data - исходные данные
len - количества исходных данных
LowFreq - нижняя граничащая частота
HighFreq - нижняя граничащая частота
samplingRate - частота дискретизации
return:
data - отфильтравынные данные
*/
float* filtering(int order, float* data, int len, Ipp64f lowFreq, Ipp64f highFreq, Ipp64f samplingRate)
{
	/* отношение граничащих частот к частоте дискретизации
	т.е. тут от 0 до 0,5 должно быть */
	double rLowFreq = (double)lowFreq / (double)samplingRate;
	double rHighFreq = (double)highFreq / (double)samplingRate;
	if (rLowFreq > 0.5 || rLowFreq < 0 || rHighFreq > 0.5 || rHighFreq < 0 || rHighFreq < rLowFreq)
	{
		printf("\n cutOffFreq is out of range");
		return NULL;
	}

	return filtering(order, data, len, rLowFreq, rHighFreq);
}

/*
parametrs:
env: the JNI interface pointer.
jClass: a Java class object.
order - поярдок фильтра
data - исходные данные
len - количества исходных данных
LowFreq - нижняя граничащая частота
HighFreq - нижняя граничащая частота
samplingRate - частота дискретизации
return:
data - отфильтраванные данные
int - номер ошибки
*/
JNIEXPORT jint JNICALL Java_RecordAndProcessing_filtering_IppFiltering_filtering
(JNIEnv *env, jclass jClass, jint order,
	jfloatArray data, jint len, jdouble rLowFreq, jdouble rHighFreq, jdouble samplingRate)
{
	/*выделение памяти*/
	jfloatArray result = env->NewFloatArray(len);
	if (result == NULL) {
		printf("\n Out of memory error");
		return 1; /* out of memory error thrown */
	}

	/*чтение входных данных*/
	jfloat* fltData = env->GetFloatArrayElements(data, NULL);
	if (fltData == NULL) {
		printf("\n Read data error");
		return 2; 
	}

	/*фильтрация*/
	filtering(order, fltData, len, rLowFreq, rHighFreq, samplingRate);
	if (fltData == NULL) {
		printf("\n Filtering error");
		return 3;
	}

	/*освобождение памяти и копирование отфильтрованных данных в data */
	/*
	data: a Java array object.
	fltData: a pointer to array elements
	mode: the release mode. 0 - copy back the content and free the elems buffer*/
	env->ReleaseFloatArrayElements(data, fltData, 0);
	return 0;
}
