/* RESEARCHER-03; dependency-free ZIP/DEFLATE reader for exact binary evidence retrieval.
   Does not execute model inference or fitting. Node 22 / V8.
*/
function read16(a,i){return a[i]|a[i+1]<<8}
function read32(a,i){return (a[i]|a[i+1]<<8|a[i+2]<<16|a[i+3]<<24)>>>0}
function binaryText(a){let s='';for(let i=0;i<a.length;i+=8192)s+=String.fromCharCode(...a.subarray(i,i+8192));return s}
const LB=[3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258];
const LE=[0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0];
const DB=[1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577];
const DE=[0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13];
function inflate(src,expected){let pos=0,bitCount=0,bitBuf=0;let output=new Uint8Array(expected),out=0;
 function read(n){while(bitCount<n){if(pos>=src.length)throw Error('Deflate input EOF');bitBuf|=src[pos++]<<bitCount;bitCount+=8;}let v=bitBuf&((1<<n)-1);bitBuf>>>=n;bitCount-=n;return v;}
 function tree(lengths){let counts=Array(16).fill(0);for(let len of lengths)if(len)counts[len]++;let next=Array(16).fill(0),code=0,max=0;for(let n=1;n<16;n++){code=(code+counts[n-1])*2;next[n]=code;if(counts[n])max=n;}let map=Array.from({length:16},()=>new Map());for(let sym=0;sym<lengths.length;sym++){let len=lengths[sym];if(len)map[len].set(next[len]++,sym);}return{map,max};}
 function symbol(t){let code=0;for(let len=1;len<=t.max;len++){code=(code<<1)|read(1);const s=t.map[len].get(code);if(s!==undefined)return s;}throw Error('Invalid Huffman symbol at '+pos);}
 let last=0;do{last=read(1);let type=read(2),lit,dist;if(type===0){let leftover=bitCount%8;if(leftover)read(leftover);let n=read(16),inverse=read(16);if((n^inverse)!==65535)throw Error('Invalid stored block');for(let i=0;i<n;i++)output[out++]=read(8);continue;}
 if(type===1){let ls=new Array(288);for(let i=0;i<288;i++)ls[i]=i<144?8:i<256?9:i<280?7:8;lit=tree(ls);dist=tree(Array(32).fill(5));}
 else if(type===2){let nl=read(5)+257,nd=read(5)+1,nc=read(4)+4;const order=[16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15];let clen=Array(19).fill(0);for(let i=0;i<nc;i++)clen[order[i]]=read(3);const ct=tree(clen);let lens=[];while(lens.length<nl+nd){let s=symbol(ct);if(s<=15)lens.push(s);else{let count=s===16?read(2)+3:s===17?read(3)+3:read(7)+11;let value=s===16?lens[lens.length-1]:0;if(s===16&&lens.length===0)throw Error('Invalid repeat');while(count--)lens.push(value);}}lit=tree(lens.slice(0,nl));dist=tree(lens.slice(nl));}
 else throw Error('Reserved Deflate block '+type);
 while(true){const s=symbol(lit);if(s<256){if(out>=expected)throw Error('output overflow');output[out++]=s;}
 else if(s===256)break;
 else {let li=s-257;if(li<0||li>=LB.length)throw Error('bad length symbol');const len=LB[li]+read(LE[li]);const ds=symbol(dist);if(ds>=DB.length)throw Error('bad distance symbol');let distance=DB[ds]+read(DE[ds]);if(distance>out)throw Error('distance beyond output');for(let j=0;j<len;j++){if(out>=expected)throw Error('output overflow backref');output[out]=output[out-distance];out++;}}
 }
 }while(!last);
 if(out!==expected)throw Error('expected '+expected+' got '+out);return output;
}
function unzip(data){let end=-1;for(let i=data.length-22;i>=Math.max(0,data.length-66000);i--)if(read32(data,i)===0x06054b50){end=i;break}if(end<0)throw Error('No ZIP end-of-central-directory');let n=read16(data,end+10),start=read32(data,end+16),p=start,entries=[];for(let i=0;i<n;i++){if(read32(data,p)!==0x02014b50)throw Error('Bad central directory');const method=read16(data,p+10),csize=read32(data,p+20),usize=read32(data,p+24),nameLen=read16(data,p+28),extraLen=read16(data,p+30),commentLen=read16(data,p+32),offset=read32(data,p+42);const name=binaryText(data.subarray(p+46,p+46+nameLen));if(read32(data,offset)!==0x04034b50)throw Error('Bad local header');const srcStart=offset+30+read16(data,offset+26)+read16(data,offset+28);const bytes=method===8?inflate(data.subarray(srcStart,srcStart+csize),usize):method===0?data.subarray(srcStart,srcStart+csize):null;if(!bytes)throw Error('Unsupported ZIP method '+method);entries.push({name,bytes,method,usize,csize});p+=46+nameLen+extraLen+commentLen;}return entries;}
function sha256(data){
 const primes=[0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];
 const l=data.length;const pad=new Uint8Array((l+9+63)&~63);pad.set(data);pad[l]=128;const bits=BigInt(l)*8n;for(let j=0;j<8;j++)pad[pad.length-1-j]=Number((bits>>BigInt(j*8))&255n);
 let h=[0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];const rr=(x,n)=>x>>>n|x<<(32-n);
 for(let p=0;p<pad.length;p+=64){const w=Array(64).fill(0);for(let i=0;i<16;i++)w[i]=(pad[p+4*i]<<24|pad[p+4*i+1]<<16|pad[p+4*i+2]<<8|pad[p+4*i+3])>>>0;for(let i=16;i<64;i++){let x=w[i-15],y=w[i-2];w[i]=(w[i-16]+(rr(x,7)^rr(x,18)^x>>>3)+w[i-7]+(rr(y,17)^rr(y,19)^y>>>10))>>>0;}
 let [a,b,c,d,e,f,g,j]=h;for(let i=0;i<64;i++){let S1=rr(e,6)^rr(e,11)^rr(e,25);let t1=(j+S1+((e&f)^(~e&g))+primes[i]+w[i])>>>0;let S0=rr(a,2)^rr(a,13)^rr(a,22);let t2=(S0+((a&b)^(a&c)^(b&c)))>>>0;j=g;g=f;f=e;e=(d+t1)>>>0;d=c;c=b;b=a;a=(t1+t2)>>>0;}let t=[a,b,c,d,e,f,g,j];for(let i=0;i<8;i++)h[i]=(h[i]+t[i])>>>0;}
 return h.map(x=>x.toString(16).padStart(8,'0')).join('');
}
globalThis.RESEARCHER03_ZIP_TOOLS={unzip,binaryText,sha256};
if(typeof module!=='undefined')module.exports=globalThis.RESEARCHER03_ZIP_TOOLS;
