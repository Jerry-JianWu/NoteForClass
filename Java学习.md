# Java学习

---

## 什么是JDK，JRE

- JDK基本介绍

JDK的全称Java Development Kit ，JDK=JRE + java的开发工具

JDK是提供给Java开发人员使用的，其中包含了java的开发工具，也包括了JRE。所以安装了JDK，就不用单独安装JRE了

- JRE 基本介绍

JRE的全称Java Runtime Environment  JRE=JVM +Java的核心库

包括Java虚拟机和Java程序所需的核心类库等，如果想要运行一个开发好的java程序，计算机中只需要安装JRE即可。



## 安装jdk8

1. 下载jdk8 64位
2. 安装开发工具，创建两个文件夹，一个安装jdk8，一个安装jre
3. 增加`JAVA_HOME`环境变量，在变量中添加JAVA_HOME，在path路径中添加`%JAVA_HOME%\bin`

![image-20231122094305489](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202311220943572.png)![image-20231122094316322](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202311220943350.png)

此时打开dos界面输入java或者javac出现命令提示则成功。

4. maca安装完jdk8后，在终端输入`source .bash_profile`即可。

## IDEA使用

在idea中run一个文件时，会先编译成`.class`文件，在运行。

### IDEA常用快捷键

1. 删除当前行，在keymap中搜索delete配置成`ctrl + d`
2. 复制当前行，在keymap中搜索duplicate改成`ctrl + alt + 向下光标`
3. 补全代码,`alt + /`
4. 添加注释和取消注释`ctrl + /`
5. 导入该行需要的类，先配置auto import，然后再使用`alt+enter`

![image-20231122111214436](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202311221112542.png)

6. 快速格式化代码`ctrl + alt + l`

7. 快速运训程序，搜索run自定义为`alt + r`
8. 生成构造器等`alt + insert` 提高开发效率
9. 查看一个类的层级关系 `ctrl + h` 学习继承后非常有用
10. 将光标放在一个方法上，输入` ctrl +b `，可以选择定位到哪个类的方法
11. 自动的分配变量名，通过在后面`.var`
12. 模板使用，template

![image-20231122113242969](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202311221132029.png)

## Java执行流程分析

![image-20240103102034657](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202401031020740.png)

![image-20240103102043065](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202401031020157.png)

1. Java源文件以`.java`为扩展名。源文件的基本组成部分是类（class）。
2. Java应用程序的执行入口是`main()`方法。有固定的书写格式`public static void main(String args){...}`
3. Java语言严格区分大小写。
4. Java方法由一条条语句构成，每个语句以`;`结束。
5. 大括号都是成对出现，缺一不可。
6. 一个源文件中最多只能有一个`public`类。其他类的个数不限
7. 如果源文件包含一个`public`类，则文件名必须按该类名命名
8. 一个源文件中最多只能有一个`public`类，其他类的个数不限，也可以将`main方法`写在`非public类`中，然后指定运行`非public类`，这样入口方法就是`非public的main方法`

## 面向对象编程（基础部分）

### 类与对象

```java
class Cat{
    // attri
    String name; // 
    int age;
    String color;
    double weight;
    
    // action
}
```

1) 类是抽象的概念，代表一类事物，是数据类型
2) 对象是具体的，实际的，代表一个具体事物，即是实例
3) 类是对象的模板，对象是类的一个个体，对应一个实例

![image-20240104165937289](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202401041659461.png)

##### 属性/成员变量/字段

基本介绍：

从概念或叫法上：成员变量=属性=field，属性是类的一个组成部分，一般是基本数据类型

属性的定义类型可以为任意类型，包含基本类型或引用类型。

##### 如何创建对象

1.先声明在创建

```java
Cat cat;
cat = new Cat();
```

2.直接创建

```java
Cat cat = new Cat();
```

##### 如何访问属性

```java
cat.name;
cat.age;
cat.color;
```

##### 类和对象的内存分配机制

![image-20240104172549725](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202401041725819.png)

![image-20240104172556189](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202401041725250.png)

###### java内存的结构分析

1）栈：一般存放基本数据类型（局部变量）

2）堆：存放对象（Cat cat，数组等）

3）方法区：常量池（常量，比如字符串等），类加载信息

4）示意图

```java
Person p = new person();
p.name = "jack";
p.age = 10;
```

1. 先加载Person类信息（属性和方法信息，只会加载一次）
2. 在堆中分配空间，进行默认初始化(看规则)
3. 把地址赋给p，p就指向对象
4. 进行指定初始化。

### 成员方法

##### 快速入门

1）添加speak成员方法，输出“我是一个好人”

2）添加cal01成员方法，可以计算从1+...+1000的结果

3）添加cal02成员方法，该方法可以接收一个数n，计算从1+...+n的结果

4）添加getSum成员方法，可以计算两个数的和。

```java
class Person{
    String name;
    int age;
    
    public void speak(){
        System.out.println("i am a good man");
    }
    
    public void cal01(){
        int res = 0;
        for(int i = 1; i<=1000; i++){
            res += i;
        }
        System.out.println("resualt of cal01 method is " + res);
    }
    
    public void cal02(int n){
         int res = 0;
        for(int i = 1; i <= n; i++){
            res += i;
        }
        System.out.println("resualt of cal02 method is " + res);
    }
    public int getSum(int num1, int num2){
        int res = num1 + num2;
        return res;
    }
    
}
public class Method01{
    public static void main(String[] args){
        Person p1 = new Person()
        p1.speak();
        p1.cal01();
        p1.cal02(5);
        p1.cal02(10);
        int returnRes = p1.getSum(10, 20);
}
```

##### 方法的调用机制原理

![image-20240105101249150](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202401051012287.png)

##### 成员方法的好处

1）提高代码的复用性

2）可以将实现的细节封装起来，然后供其他用户来调用即可

##### 成员方法传参机制（极其重要）

###### 对于基本数据类型

基本数据类型，传递的是值（值拷贝），形参的任何改变不影响实参。（不影响外部数据）

##### 对于引用数据类型的传参机制

引用类型传递的是地址（传递的也是值，但是值是地址），可以通过形参来影响实参。

###### 成员方法返回类型是引用类型应用实例

1）编写类MyTools类，编写一个方法可以打印二维数组的数据

2）编写一个方法copyPerson，可以复制一个Person对象，返回复制的对象。克隆对象，注意要求得到新对象和原来的对象是两个独立的对象，只是他们的属性相同。

```java
public class MethodExercise02 {
    public static void main(String[] args) {
        Person p1 = new Person();
        p1.age = 10;
        p1.name = "zhangsan";
        Person p2 = new Person();
        MyTools mytools = new MyTools();
        p2 = mytools.copyPerson(p1);
        System.out.println(p2.age);
    }
}
class MyTools {
    // copy obj person
    //编写方法的思路
    //1. 方法的返回类型 Person
    //2. 方法的名字 copyPerson
    //3. 方法的形参 (Person p)
    //4. 方法体, 创建一个新对象，并复制属性，返回即可
    public Person copyPerson(Person p) {
        Person p2 = new Person();
        p2.age = p.age;
        p2.name = p.name;
        return p2;
    }

}

class Person {
    int age;
    String name;

}
```

### 方法重载OvaerLoad

java中允许同一个类中，多个同名方法的存在，但要求形参列表不一致

#### 快速入门案例

类：MyCalculator，方法：calculate

1)calculate(int n1, int n2) //两个整数的和
	 2)calculate(int n1, double n2) //一个整数，一个 double 的和
	 3)calculate(double n2, int n1)//一个 double ,一个 Int 和
	 4)calculate(int n1, int n2,int n3)//三个 int 的和

```java
public class Overload01 {
    public static void main(String[] args) {
        MyCalculate myCalculate = new MyCalculate();
        System.out.println(myCalculate.calcute(1, 2));
        System.out.println(myCalculate.calculate(1, 2.0));
        System.out.println(myCalculate.calculate(1.0, 2));
        System.out.println(myCalculate.calculate(1, 2, 3));

    }
}
class MyCalculate{
    public int calcute(int n1, int n2){
        System.out.println("int + int is called");
        return n1 + n2;
    }
    public double calculate(int n1, double n2){
        System.out.println("int + double is called");
        return n1 + n2;
    }
    public double calculate(double n1, int n2){
        System.out.println("double + int is called");
        return n1 + n2;
    }
    public int calculate(int n1, int n2, int n3){
        System.out.println("int + int + int is called");
        return n1 + n2 + n3;
    }
}

```

#### 注意事项和使用细节

1）方法名：必须相同

2）形参列表：必须不同

3）返回类型：无要求



### 可变参数

java允许将同一个类中多个同名同功能但参数个数不同的方法，封装成一个方法。就可以通过可变参数实现。

基本语法：

```java
访问修饰符 返回类型 方法名(数据类型... 形参名){}
```

##### 快速入门案例

类HspMethod，方法sum：可以计算两个数的和，三个数的和等等

```java
public class VarParameter01 {
    public static void main(String[] args) {
        varMethod var = new varMethod();
        System.out.println(var.sum(1,5,100));
        System.out.println(var.sum(1, 19, 20, 30, 40, 50));

    }
}
class varMethod{
    public int sum(int... nums){
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
        }
        return sum;
    }
}

```

### 构造方法/构造器

 基本语法：

```java
[修饰符] 方法名(形参列表){
  方法体}
```

构造方法又叫构造器，是类的一种特殊的方法，它的主要作用是完成对新对象的初始化。

特点：

1. 方法名和类名相同
2. 没有返回值
3. 在创建对象时，系统会自动的调用该类的构造器完成对象的初始化。



```java
public class Constructor01{
  public static void main(String[] args){
    //new 对象,直接通过构造器制定名字和年龄
    Person p1 = new Person("smith", 80);
    System.out.println("p1 的信息如下");
    System.out.println("p1 对象name=" + p1.name);//smith
    System.out.println("p1 对象age=" + p1.age);//80
  }
}

class Person{
  String name;
  int age;
  //1.构造器没有返回值，不能写void
  //2.构造器的名称和类Person一样
  //
  public Person(String pName, int pAge){
    System.out.println("构造器被调用 完成对象的属性初始化");
    name = pName;
    age = pAge;
  }
}
```

#### 注意事项和使用细节

1. 一个类可以定义多个不同的构造器，即构造器重载
2. 构造器名和类名要相同
3. 构造器没有返回值
4. 构造器是完成对象的初始化，并不是创建对象
5. 在创建对象时，系统自动的调用该类的构造方法
6. 如果程序员没有定义构造器，系统会自动给类生成一个默认无参构造器
7. 一旦定义了自己的构造器，默认的构造器就覆盖了，不能再使用默认的无参构造器。除非显式地重新定义无参构造器



### this 关键字

案例

```java
class Dog{
  public String name;
  public int age;
  public Dog(String dName, int dAge){
    name = dName;
    age = dAge;
  }
  public void info(){
    System.out.println(name + "\t" + age + "\t");
  }
}
```

问题：构造方法的输入参数名不是非常好，如果能将dName改成name就好了，但是我们会发现按照变量的作用域原则，name 的值就是null，怎么解决，使用this

#### 什么是this

1. 区分实例变量和参数名：当方法的参数名与类的实例变量名相同，`this`关键字可以用来区分实例变量和参数名。

   ```java
   public class Example {
       private int value;
       
       public void setValue(int value) {
           this.value = value; // 使用this来指明实例变量
       }
   }
   
   ```

2. 在构造器中调用另一个构造器：可以使用`this()`语法在一个构造器中调用另一个构造器，通常用于构造器重载的情况，以避免代码重复.

   ```java
   public class Example {
       private int x, y;
       
       public Example() {
           this(0, 0); // 调用另一个构造器
       }
       
       public Example(int x, int y) {
           this.x = x;
           this.y = y;
       }
   }
   
   ```

3. 返回当前类的实例：`this`关键字可以在方法中使用，以返回当前类的实例。

   ```java
   public class Example {
       public Example get() {
           return this; // 返回当前实例
       }
   }
   
   ```

4. 传递当前对象作为参数：有时需要将当前对象传递给另一个方法或在当前类的上下文中使用，`this`关键字可以用于这种情况。

   ```java
   public class Example {
       public void print() {
           anotherMethod(this); // 将当前对象作为参数传递
       }
       
       public void anotherMethod(Example obj) {
           // 使用obj
       }
   }
   
   ```

   

## 面向对象编程（中级部分）

### 包

1. 区分相同名字的类
2. 类很多时，可以很好的管理类
3. 控制访问范围

包的本质实际上是创建不同的文件夹/目录来保存类文件

### 访问修饰符

java提供四种访问控制修饰符，用于控制方法和属性的访问权限：

1）公开级别：public，对外公开

2）受保护级别：protect，对子类和同一个包中的类公开（不同包）

3）默认级别：没有修饰符号，向同一个包的类公开（不同类不同包）

4）私有级别：用private修饰，只有类本身可以访问，不对外公开

注意事项：

1）修饰符可以用来修饰类中的属性，成员方法及类

2）只有默认的和public才能修饰类，并且遵循上述访问权限的特点

3）成员方法的访问规则和属性完全一样





## 面向对象编程三大特征

### 封装

封装：就是把抽象出的数据[属性]和对数据的操作[方法]封装在一起，数据被保护在内部，程序的其他部分只有通过被授权的操作[方法]，才能对数据进行操作。

封装的理解和好处：

1）隐藏实现细节：方法（连接数据库）<-调用（传入参数）

2）可以对数据进行验证，保证安全合理

封装的实现步骤

1）将属性进行私有化`private`

2）提供一个公共的(public)set方法，用于对属性判断并赋值

3）提供一个公共的get方法，用于获取属性的值

### 继承

继承可以解决代码复用，让我们的编程更加靠近人类思维，当多个类存在相同的属性和方法时，可以从这些类中抽象出父类，在父类中定义这些相同的属性和方法，所有的子类不需要重新定义这些属性和方法，只需要通过`extends`来声明继承父类即可。

基本语法

```java
class son extends father{
  
}
```

1）子类继承了所有的属性和方法，非私有的属性和方法可以在子类直接访问，但是私有属性和方法**不能在子类直接访问**，要通过父类提供的公共的方法去访问。

2）子类必须要调用父类的构造器，完成父类的初始化

3）当创建子类对象时，不管使用子类的哪个构造器，默认情况下总会去调用父类的无参构造器，如果父类没有提供无参构造器，则必须在子类的构造器中用super取指定使用父类的哪个构造器完成对父类的初始化工作，否则，编译不会通过。

4）如果希望指定调用父类的某个构造器，则显式的调用一下：`super(parameter_list)`

5）super在使用时，必须放在构造器第一行（super只能在构造器中使用）

6）`super()` 和`this()`都只能放在构造器第一行，因此这两个方法不能共存在一个构造器

7）java所有类都是Object类的所有子类，Object类是所有类的基类

8）父类构造器的调用不限于直接父类将一直往上追溯到Object类

9）子类最多只能继承一个父类（直接继承），即java中是单继承机制。

10）不能滥用继承，子类和父类之间必须满足`is-a`的逻辑关系

### super关键字

super代表父类的引用，用于访问父类的属性、方法、构造器

基本语法

```java
1.访问父类的属性，但不能访问父类的private
  super.attribute;
2.访问父类的方法，不能访问父类的private方法
  super.method(paramlist);
3.访问父类的构造器：
  super(paramlist);只能放在构造器的第一句
```

#### super给编程带来的便利/细节

1.调用父类的构造器的好处

2.当子类中有和父类汇总成员重名时，为了访问父类的成员，必须通过super。如果没有重名，使用super、this、直接访问是一样的效果。

3.super的访问不限于直接父类，如果爷爷类和本类中有同名的成员，也可以使用super去访问爷爷类的成员；如果多个基类都有同名的成员，使用super访问遵循就近原则。`A->B->C`，当然也需要遵守访问权限的相关规则。

| No   |   区别点   |                          this                          |                  super                   |
| ---- | :--------: | :----------------------------------------------------: | :--------------------------------------: |
| 1    |  访问属性  | 访问本类中的属性，如果本类没有此属性则从父类中继续查找 |            从父类开始查找属性            |
| 2    |  调用方法  |    访问本类中的方法，如果没有此方法则从父类继续查找    |            从父类开始查找方法            |
| 3    | 调用构造器 |          调用本类构造器，必须放在构造器的首行          | 调用父类构造器，必须放在子类构造器的首行 |
| 4    |    特殊    |                      表示当前对象                      |            子类中访问父类对象            |

### 方法重写/覆盖（override）

简单的说：方法覆盖（重写）就是子类中有一个方法，和父类的某个方法的名称、返回类型、参数一样，那么我们就说子类的这个方法覆盖了父类的方法。

主要事项和使用细节：

1. 子类的方法的形参列表，方法名称，要和父类方法的形参列表，方法名称完全一样。
2. 子类方法的返回类型和父类方法返回类型是一样，或者是父类返回类型的子类。
3. 子类方法不能缩小父类的访问权限。





### 多态

多态：允许对象以其所属的类的方式被视为是从其他类派生的。简而言之，多态性允许一个接口（通常是一个方法或函数）被多种不同的数据类型的对象所共用。这意味着不同的对象可以通过实现相同的接口以各自的方式响应相同的消息。**[允许在不同的对象上调用相同的方法，而这些对象可以通过各自的方法来响应同一个消息]**

1. 子类（继承）多态：通过继承，子类可以**重写**父类的方法，使得即使是通过父类的引用调用该方法时，也会执行子类的方法实现。体现`is-a`的关系
2. 接口多态：类可以实现（implement）一个或多个接口，而接口仅定义方法的签名，不实现方法。实现接口的类必须提供接口中所有方法的具体实现。

重要：

1）一个对象的编译类型和运行类型可以不一致

2）编译类型在定义对象时，就确定了，不能改变

3）运行类型是可以变化的

4）编译类型看定时=号左边，运行类型看=号的右边

#### 多态快速入门案例

使用多态的机制来解决宠物主人喂食物的问题。

```java
public void feed(Animal animal, Food food){
  System.out.println("animal" + animal + "food" + food);
  System.out.println(animal.getName() + "eat" + food.getName());
}

```

多态事项和细节讨论：

多态的前提是：两个对象存在继承关系

多态的**向上转型**：

向上转型是将子类的引用自动转换为父类引用的过程。过程自动安全。

1）本质：父类的引用只想了子类的对象

2）语法：`父类类型 引用名=new 子类类型();`

3）特点：编译类型看左边，运行类型看右边。

```java
class Animal:
    def makeSound(self):
        print("Some generic sound")

class Dog(Animal):
    def makeSound(self):
        print("Woof")

class Cat(Animal):
    def makeSound(self):
        print("Meow")

# 使用多态
animals = [Dog(), Cat(), Animal()]

for animal in animals:
    animal.makeSound()

```



多态的**向下转型**：

向下转型是将父类的引用转换为子类引用的过程，过程不是自动的，需要显式地进行，需要类型检查，以确保安全

1）语法：`子类类型 引用名 = (子类类型) 父类引用；`

2）只能强转父类的引用，不能强转父类的对象

3）要求父类的引用必须指向的是当前目标类型的对象

4）当向下转型后，可以调用子类类型中所有成员

```java
class Animal {}
class Dog extends Animal {
    void bark() {
        System.out.println("Woof");
    }
}

// 向上转型
Animal a = new Dog();

// 向下转型
if (a instanceof Dog) { // 检查类型
    Dog d = (Dog) a;
    d.bark(); // 现在可以调用 Dog 类的方法了
}
在这个例子中，a 首先通过向上转型被赋值为 Dog 类型的对象。然后，我们通过 instanceof 关键字检查 a 是否真的是 Dog 的一个实例，以确保向下转型是安全的，之后才执行向下转型并调用 Dog 类特有的 bark 方法。

```

### Object类详解

#### ==和equals对比

==是一个比较运算符

1. ==：既可以判断基本类型，又可以判断引用类型。
2. ==：如果判断基本类型，判断的是值是否相等。
3. ==：如果判断引用类型，判断的是地址是否相等，即判定是不是同一个对象。
4. equals：Object类中的方法，只能判断引用类型
5. 默认判断的是地址是否相等，子类中往往重写该方法，用于判断内容是否相等。

#### hasCode方法

`public int hasCode()`

返回该对象的哈希值，支持此方法是为了提高哈希表的性能。哈希值主要根据地址号来，不能完全将哈希值等价于地址。

两个引用指向同一个对象，哈希值相同，指向不同对象，哈希值不同。

#### toString方法

默认返回：`全类名+@+哈希值的十六进制`

子类往往重写toString方法，用于返回对象的属性信息。

#### finalize方法

1）当对象被回收时，系统自动调用该对象的finalize方法。子类可以重写该方法，做一些释放资源的操作

2）什么时候被回收：当某个对象没有任何引用时，则jvm就认为这个对象是一个垃圾对象，就会使用垃圾回收机制来销毁该对象，销毁对象前，会先调用finalize方法。

3）垃圾回收机制的调用，是由系统来决定的。实际开发几乎不会用finalize，所以更多就是为了面试。



## 面向对象编程（高级部分）

### 类变量

类变量：也叫静态变量/静态属性，是该类的所有对象共享的变量，任何一个该类的对象去访问它时，取到的都是相同的值，同样任何一个该类的对象去修改它时，修改的也是同一个变量

定义语法：`访问修饰符 static 数据类型 变量名;`

如何访问类变量：`类名.类变量名 or 对象名.类变量名`

#### 类变量使用的注意事项和细节讨论：

1.什么时候需要使用类变量

当我们需要让某个类的所有对象都共享一个变量时，就可以考虑使用类变量

2.类变量与实例变量区别

类变量是该类的所有对象共享的，而实例变量是每个对象独享的

3.加上static成为类变量或静态变量，否则成为实例变量/普通变量

4.实例变量不能通过`类名.类变量名`方式访问

5.类变量可以通过`类名.类变量名`来访问

6.类变量是在类加载时就初始化了，也就是说，即使没有创建对象，只要类加载了，就可以使用类变量了

7类变量的生命周期是随类的加载开始，随着类消亡而销毁

### 类方法

类方法也叫静态方法

`访问修饰符 static 数据返回类型，方法名(){}`

类方法的调用：

`类名.类方法名`

#### 类方法的经典使用场景及注意事项和细节讨论

当方法中不涉及到任何和对象相关的成员，则可以将方法设计成静态方法，提高开发效率

1）类方法和普通方法都是随着类的加载而加在，将结构信息存储在方法去：类方法中无this的参数，普通方法中隐含着this参数

2）类方法可以通过类名调用，也可以通过对象名调用

3）普通方法和对象有关，需要通过对象名调用，比如`对象名.方法名`

4）类方法中不允许使用和对象有关的关键字，比如this和super，普通方法可以

5）类方法中只能访问静态变量和静态方法

6）普通成员方法，既可以访问非静态成员，也可以访问静态成员

### 理解main方法语法

main方法的形式：`public static coid main(String[] args){}`

1.main方法是虚拟机调用

2.java虚拟机需要调用类的`main()`方法，所以该方法的访问权限必须是public

3.jvm在执行main方法时不必创建对象，所以该方法必须是static

4.该方法接受String类型的数组参数，该数组中保存执行java命令时传递给所运行的类的参数

特别提示：

1.在`main()`方法中，我们可以直接调用main方法所在类的静态方法和静态属性

2）但是，不能直接访问该类中的非静态成员，必须创建该类的一个实例对象后，才能通过这个对象去访问类中的非静态成员

### 单例设计模式

1.静态方法和属性的经典使用

2.设计模式时在当量的实践中总结和理论话之后优选的代码结构、编程风格以及解决问题的思考方式。

单例模式：采取一定的方法保证在整个软件系统中，对某个类只能存在一个对象实例，并且该类只提供一个取得其对象实例的方法

**单例模式有两种方式**：1）饿汉式。2）懒汉式



单例模式应用关键点：

1. 私有化构造函数：确保外部不能直接实例化
2. 一个类变量：存储唯一实例的引用
3. 向外暴露一个公共的静态方法：提供全局访问点，并负责创建和返回唯一实例

#### 懒汉式

```java
public class SingletonLazyThreadSafe {
    private static SingletonLazyThreadSafe instance;

    private SingletonLazyThreadSafe() {}

    public static synchronized SingletonLazyThreadSafe getInstance() {
        if (instance == null) {
            instance = new SingletonLazyThreadSafe();
        }
        return instance;
    }
}

```

#### 饿汉式

```java
public class SingletonEager {
    private static final SingletonEager instance = new SingletonEager();

    private SingletonEager() {}

    public static SingletonEager getInstance() {
        return instance;
    }
}

```

#### 饿汉式与懒汉式区别

1. 二者最主要的区别在于创建对象的实际不同：饿汉式是在类加在就创建了对象实例，而懒汉式是在使用时才创建
2. 饿汉式不存在线程安全问题，懒汉式存在线程安全问题。
3. 饿汉式存在浪费资源的可能。因为如果程序员一个对象实例都没有使用，那么饿汉式创建的对象就浪费了，懒汉式式使用时才创建，就不存在这个问题。



### final 关键字

基本介绍：

在某些情况下，程序员可能有以下需求，会使用到final：

1. 当不希望类被继承时，可以用final修饰
2. 当不希望父类的某个方法被子类覆盖/重写时，可以用final关键字修饰
3. 当不希望类的某个属性的值被修改，可以用final修饰
4. 当不希望某个局部变量被修改，可以使用final修饰

#### final 使用注意事项和细节讨论

1. final修饰的属性又叫常量

2. final修饰的属性在定义时，必须赋初值，并且以后不再修改。
3. 如果final修饰的属性是静态的，则初始化位置只能是1）定义时。2）在静态代码块中，不能再构造器中国呢
4. final类不能继承，但是可以实例化对象
5. 如果类不是final类，但是含有final方法，则该方法虽然不能重写，但是可以被继承。
6. 一般来说如果一个类已经是final类了，就没有必要再将方法修成final方法
7. final方法不能修饰构造方法
8. final和static往往搭配使用，效率更高

### 抽象类

当父类的某系方法，需要声明，但是又不确定如何实现时，可以将其声明为抽象方法，那么这个类就是抽象类，使用`abstract`关键字来修饰该方法，这个方法就是抽象方法，用abstract来修饰该类就是抽象类。

### 抽象类的使用注意事项和细节讨论

1）抽象类不能被实例化

2）抽象类不一定要包含abstract方法

3）一旦类包含了abstract方法，则这个类必须声明为abstract

4）abstract只能修饰类和方法，不能修饰属性和其他的

5）抽象类可以有任意成员

6）抽象方法不能有主体，即不能实现

7）如果一个类继承了抽象类，则它必须实现抽象类的所有抽象方法，除非它自己也声明为abstract类。

9）抽象方法不能使用private、final和static来修饰，因为这些关键字都是和重写相违背的。

### 接口Interface

接口是一种引用类型，它是完全抽象的类，即它不能有任何的实现（方法体），只能有声明的方法和公共的静态常量。接口是用来建立类与类之间的协议。一个类通过实现接口的方式，从而继承接口中的抽象方法。

接口就是给出一些没有实现的方法，封装到一起，到某个类要使用的时候，在根据具体情况把这些方法写出来。主要作用有：

1. 定义规则：接口定义了某一批类所需要遵守的规则，接口中的方法定义乐类的行为
2. 实现多重继承：Java不支持多继承（一个类继承多个类），但通过实现多个接口，一个类可以继承多个接口中的方法，这种方式可以间接实现多继承。
3. 实现解耦：接口可以用来实现解耦合，使得代码更加模块化，提高代码的可维护性和可扩展性。

语法：

```java
interface name{
	// arttribute
    // abstract method
}
class classname implements interface{
    自己属性；
    自己方法;
    必须实现的接口的抽象方法
}
小结：接口是更加抽象的抽象的类，抽象类里的方法可以有方法体，接口里的所有方法都没有方法体[jdk7.0]
    接口体现了程序设计的多态和高内聚低耦合的设计思想。
    特别说明：Jdk8.0后接口类可以有静态方法，默认方法，也就是说接口中可以有方法的具体实现
```

一些具象化的理解：

1. ​    比如说要制造战斗机，武装直升机，专家”只需要“把飞机需要的功能/规格定下来即可，具体的实现“交给别人”做
2. 项目经理管理三个程序员，开发一个软件，为了控制和管理软件，项目经理定义一些接口，然后由程序员具体实现

假设有一个接口“Animal”，定义了所有动物类的必须实现的方法‘eat’和’sleep’。

```java
interface Animal{
    void eat();
    void sleep();
}
class Dog implements Animal{
    public void eat() {
        System.out.println("Dog is eating");
    }
    public void sleep() {
        System.out.println("Dog is sleeping");
    }
}

class Cat implements Animal{
    public void eat() {
        System.out.println("Cat is eating");
    }
    public void sleep() {
        System.out.println("Cat is sleeping");
    }
}
```

### 内部类

Java中的内部类是定义在另一个类里面的类。内部类提供了一种强大的编程机制，可以让你逻辑地分组类和接口，以便更好地组织代码，并提供一种方式来封装与外部类相关的功能而不必使其堆外部世界可见。

主要分为四种类型：

1. 成员内部类：最普通的内部类，它需要被实例化后才能使用，不能含有静态方法和静态变量（除非声明为final且为编译时常量）。它可以访问外部类的所有成员（包括私有）。
2. 静态内部类（也成为静态嵌套类）：它是static定义的内部类，可以不依赖于外部类实例被实例化，并且它只能访问外部类的静态成员和方法。
3. 局部内部类：定义在方法中的内部类，它旨在该方法的执行期间存在，不能有访问修饰符和static修饰符，但可以访问方法内的局部变量（局部变量必须是final或effectively final）。
4. 匿名内部类：没有名称的局部内部类，用于临时创建一个继承自类或接口的子类的对象。通常用于实现接口或抽象类的一次性使用。

```java
class OuterClass {
    private static String staticVar = "Static Variable";
    private String instanceVar = "Instance Variable";

    // 成员内部类
    class InnerClass {
        void display() {
            System.out.println(instanceVar); // 访问外部类的实例变量
        }
    }

    // 静态内部类
    static class StaticNestedClass {
        void display() {
            System.out.println(staticVar); // 只能访问外部类的静态变量
        }
    }

    void outerMethod() {
        final int num = 23;

        // 局部内部类
        class LocalInnerClass {
            void display() {
                System.out.println("Local variable num is " + num);
            }
        }
        
        LocalInnerClass lic = new LocalInnerClass();
        lic.display();
    }
}

public class Test {
    public static void main(String[] args) {
        // 实例化成员内部类
        OuterClass outer = new OuterClass();
        OuterClass.InnerClass inner = outer.new InnerClass();
        inner.display();

        // 实例化静态内部类
        OuterClass.StaticNestedClass staticInner = new OuterClass.StaticNestedClass();
        staticInner.display();

        // 调用外部类方法，该方法中包含局部内部类
        outer.outerMethod();

        // 匿名内部类，实现Runnable接口
        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                System.out.println("Anonymous Inner Class");
            }
        };
        runnable.run();
    }
}

```

## 枚举与注解

### 枚举enum

枚举是一种特殊的类，用于表示一个固定的常量集合。在Java中，枚举用`enum`关键字声明。使用枚举可以使代码更加易于阅读和维护，因为它们可以有自己的方法和构造函数。枚举类型的每一个元素都是枚举的一个实例，并且它们都是公开的、静态的和最终的（public, static, final）。从Java 5开始引入枚举，以提供一种更安全、更易用的方式来定义一组常量，相比于早期Java版本中使用`public static final`字段来定义常量的做法。

理解：一种特殊的类，里面只包含一组**“有限的特定的”**对象

```java
enum Color{
    Red, Green, Blue;
}
```

注意事项：

1. 当使用enum关键字开发一个枚举类时，默认会继承Enum类，而且是一个final类
2. 如果使用无参构造器创建枚举对象，则实参列表和小括号都可以省略
3. 当有多个枚举对象时，使用`,`间隔，最后有一个分号结尾
4. 枚举对象必须放在枚举类的行首

### 注解 Annotation

注解是Java提供的一种元数据形式，用于在代码中添加信息，这些信息可以在编译时、类加载时或运行时被读取，并且可以影响程序的行为。注解不直接影响代码的操作，但可以被用于给编译器、开发工具或运行时库提供信息。Java的注解可以用于类、方法、变量、参数和Java包等。从Java 5开始引入注解。

注解的主要用途包括：

- 编译检查：如`@Override`注解，表示子类方法覆盖了父类方法。
- 编译时和部署时的处理：如处理注解生成额外的源代码或XML文件。
- 运行时处理：某些注解可以通过反射在运行时被查询到，从而影响运行时行为。

```java
@Override
public String toString() {
    return "Example{}";
}

@SuppressWarnings("unchecked")
public void myMethod() {
}

```

`@Override`告诉编译器你打算重写一个父类或接口中的方法，而`@SuppressWarnings`用于告诉编译器忽略特定的警告（在这个例子中是未检查的类型转换）。

枚举和注解都是Java 5中引入的重要特性，它们增加了Java语言的表达能力，提高了代码的安全性和可读性。

## 异常-Exception

### 异常介绍

- 基本概念：Java语言中，将程序执行中发生的不正常情况称为“异类”。（开发过程中的语法错误和逻辑错误不是异常）
- 执行过程中所发生的异常事件可以分为两大类：

1）Error（错误）：Java虚拟机无法解决的严重问题。如：JVM系统内部错误、资源耗尽等严重情况。

2）Exception：其它因变成错误或偶然的外在因素导致的一般性问题，可以使用针对性的代码进行处理。例如空指针访问，试图读取不存在的文件，网络连接中断等等。Exception分两大类：**运行时异常**[程序运行时，发生的异常]和**编译时异常**[编程时，编译器检查出的异常]。

### 异常处理

1. `try-catch-finally`

- **捕获异常**：使用`try-catch`语句块可以捕获并处理异常。可以有多个`catch`块来捕获不同类型的异常。

![image-20240226192552417](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202402261925534.png)





2. `throws`

- **抛出异常**：使用`throw`关键字手动抛出一个异常实例。方法声明中可以使用`throws`关键字来声明该方法可能抛出的异常类型。通过`throw`关键字，你可以在代码的任何地方根据条件主动抛出异常；通过`throws`关键字，在方法声明中指出该方法可能抛出的异常类型，这是一种告知方法调用者“这个方法可能会失败，你必须准备好处理这种失败”的方式。这两种机制共同构成了Java异常处理的基础，使得异常管理更加清晰和规范。

1）如果一个方法中的语句执行时可能生成某种异常，但是并不能确定如何处理这种异常，则此方法应显式地声明抛出异常，表明该方法将不对这些异常进行处理，而由该方法的调用者负责处理。

2）在方法声明中用throws语句可以声明抛出异常的列表，throws后面的异常类型可以是方法中产生的异常类型，也可以是它的父类。

注意事项及使用细节：

1）对于编译异常，程序中必须处理，比如`try-catch ` or `throws`

2）对于运行时异常，程序中如果没有处理，默认就是`throws`的方式处理

3）子类重写父类的方法时，堆抛出的异常的规定：子类重写的方法，所抛出的异常类型要么和父类抛出的异常一直，要么是父类抛出的异常的类性的子类型。

4）在`throws`过程中，如果有方法`try-catch`，就相当于处理异常，可以不必`throws`

## 集合 Collections

### 集合的理解与好处

1）可以**动态保存**任意多个对象，使用比较方便

2）提供了一系列方便的操作对象的方法：add、remove、set、get等等

3）使用集合添加，删除新元素简洁明了

### 集合的框架体系

java集合类很多，主要分为两大类：

![image-20240226194521371](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202402261945454.png)

![image-20240226194542517](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202402261945583.png)

### Collection接口和常用方法

1）collection实现子类可以存放多个元素，每个元素可以是Object

2）有些Collection的实现类，可以存放重复的元素，有些不可以

3）有些Collection的实现类，有些是有序的（List），有些不是有序的（Set）

4）Collection接口没有直接的实现子类，是通过它的子接口Set和List来实现的

#### Collection接口遍历元素方式1-使用迭代器Iterator

1）Iterator对象称为迭代器，主要用于遍历Collection集合中的元素

2）所有实现乐Collection接口的集合类都有一个`iterator()`方法，用于返回一个实现了Iterato接口的对象，即可以返回一个迭代器。

3）Iterator仅用于遍历集合，Iterator本身并不存放对象

```java
Collection<String> collection = Arrays.asList("Apple", "Banana", "Cherry");
Iterator<String> iterator = collection.iterator();
while (iterator.hasNext()) {
    String element = iterator.next();
    System.out.println(element);
}
```

#### Collection接口遍历元素方式2-使用增强的for循环

```java
Collection<String> collection = Arrays.asList("Apple", "Banana", "Cherry");
for (String element : collection) {
    System.out.println(element);
}
```

#### Collection接口遍历元素方式3-使用java8级以上的StreamAPI

```java
Collection<String> collection = Arrays.asList("Apple", "Banana", "Cherry");
collection.stream().forEach(System.out::println);
```

#### Collection接口遍历元素方式4-使用forEach方法

从Java 8开始，`Iterable`接口添加了`forEach`方法，允许你更简洁地遍历集合，这也是使用Lambda表达式的一种方式。

```java
Collection<String> collection = Arrays.asList("Apple", "Banana", "Cherry");
collection.forEach(element -> System.out.println(element));
```

### List接口和常用方法

List接口是Collection接口的子接口

1）List集合类中元素有序（即添加顺序和取出顺序一致）、且可重复

2）List几何中的每个元素都有其对应的顺序索引，即支持索引。

3）List容器中的元素都对应一个整数型的序号记载其在容器中的位置，可以根据序号存取容器中的元素

- `ArrayList`：基于动态数组的实现，提供了快速的随机访问能力。
- `LinkedList`：基于链表的实现，优化了插入和删除操作。

#### 示例1：创建：`ArrayList`并田间元素

```java
List<String> fruits = new ArrayList<>();
fruits.add("Apple");
fruits.add("Banana");
fruits.add("Cherry");
System.out.println(fruits);
```

#### 示例2：使用`LinkedList`，在指定位置插入和删除元素

```java
List<String> animals = new LinkedList<>();
animals.add("Dog");
animals.add("Cat");
animals.add(1, "Rabbit"); // 在索引为1的位置插入“Rabbit”
System.out.println(animals);

animals.remove("Dog");
System.out.println(animals);
```

#### 示例 3: 使用`set`方法修改`ArrayList`中的元素

```java
List<String> books = new ArrayList<>();
books.add("Book1");
books.add("Book2");
books.add("Book3");

books.set(1, "New Book2"); // 将索引为1的元素替换为New Book2
System.out.println(books);
```

#### 示例 4: 使用`get`方法访问`ArrayList`中的元素

```java
List<String> cities = new ArrayList<>();
cities.add("New York");
cities.add("London");
cities.add("Tokyo");

String city = cities.get(1); // 访问索引为1的元素
System.out.println("The city at index 1 is: " + city);

```

#### 示例 5: 使用`indexOf`和`lastIndexOf`方法

```java
List<String> items = new ArrayList<>();
items.add("item1");
items.add("item2");
items.add("item1");

int firstIndex = items.indexOf("item1"); // 返回第一个"item1"的索引
int lastIndex = items.lastIndexOf("item1"); // 返回最后一个"item1"的索引
System.out.println("First index of 'item1': " + firstIndex);
System.out.println("Last index of 'item1': " + lastIndex);

```

#### 示例 6: 使用`subList`方法获取子列表

```java
List<Integer> numbers = new ArrayList<>();
for (int i = 1; i <= 5; i++) {
    numbers.add(i); // 添加数字1到5
}

List<Integer> subList = numbers.subList(1, 3); // 获取索引1（包含）到3（不包含）的子列表
System.out.println(subList);

```

#### 三种遍历方式-Iterator，增强for，普通for

```java
eg1:
Iterator iter = col.iterator();
while(iter.hasNext()){
    Object o = iter.netxt();
}

eg2:
for(Object o:col){
    
}

eg3:
for(int i=0; i<list.size();i++){
    Object object = list.get(i);
    System.out.println(object);
}
```

### ArrayList 底层结构和源码分析

#### ArrayList注意事项

1）permits all elements, including null, ArrayList可以加入null，并且可以加入多个

2）ArrayList是由数组实现数据存储的

3）ArrayList基本等同Vector，除了ArrayList是线程不安全（执行效率高），在多线程情况下不建议使用ArrayList

#### ArrayList的底层操作机制源码分析（重点，难点）

1）ArrayList重维护了一个Object类型的数组elementData。

`transient Object[] elementData; // transient表示瞬间，表示该属性不会被序列号`

2）当创建ArrayList对象时，如果使用的是无参构造器，则初始elementData容量为0，第1次添加，则扩容elementData为10，如需再次扩容，则扩容为1.5倍

3）如果使用的是指定大小的构造器，则初始elementData容量为指定大小，如果需要扩容，则直接扩容elementData为1.5倍

### Vector底层结构和源码剖析

#### Vector基本介绍

1）Vector底层也是一个对象数组，`protected Object[] elmentData;`

2）Vector是线程同步的，即线程安全

3）开发中需要线程同步安全时，考虑使用Vector



#### Vecotr 和 ArrayList的比较

![image-20240227092418251](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202402270924358.png)

### LinkedList底层结构

#### LinkedList的全面说明（基于链表实现）

1）LinkedList底层实现了双向链表和双端队列的特点

2）可以添加任意元素（元素可以重复），包括null

3）线程不安全，没有实现同步

#### LinkeList的底层操作机制

1）LinkedList底层维护了一个双向链表

2）LinkedList重维护了两个属性`first`和`last`分别指向首节点和尾节点

3）每个节点（Node对象），里面又维护了prev、next、item三个属性，其中通过prev指向前一个，通过next指向后一个节点。最终实现双向链表。

4）LinkedList元素的添加和删除，不是通过数组完成的，相对来说效率较高

#### LinkedList的增删改查案例

```java
LinkedList<String> list = new LinkedList<>();

// 增加元素
list.add("A");
list.addFirst("B");
list.addLast("C");
list.add(1, "D"); // 添加元素到索引1的位置

System.out.println("After additions: " + list);

// 删除元素
list.remove("B");
list.removeFirst();
list.removeLast();

System.out.println("After deletions: " + list);

// 修改元素
list.set(0, "E");

System.out.println("After modification: " + list);

// 查询元素
String firstElement = list.getFirst();
String lastElement = list.getLast();
System.out.println("First element: " + firstElement + ", Last element: " + lastElement);

```

#### ArrayList和LinkedList的比较

![image-20240227094059521](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202402270940650.png)

### Set接口和常用方法

#### Set接口基本介绍

1）无序（添加和取出的顺序不一致，没有索引）

2）不允许重复元素，所以最多包含一个null

#### 常用方法

和List接口一样，Set接口也是Collection的子接口，因此常用方法和Collection接口一样

#### Set接口的遍历方式

同Collection的遍历方式一样，1）可以使用迭代器，2）增强for，3）不能使用索引的方式来获取





### Set接口实现类-HashSet

#### HashSet的全面说明

1）HashSet实现了Set接口

2）HashSet实际上是HashMap

3）可以存放null值，但是只能有一个null

4）HashSet不保证元素是有序的，取决与hash后，再确定索引的结果

5）不能有重复元素/对象

#### HashSet案例说明

```java
//4 Hashset 不能添加相同的元素/数据?
set.add("lucy");//添加成功
set.add("lucy");//加入不了
set.add(new Dog("tom"));//OK
set.add(new Dog("tom"));//Ok
set.add(new String("hsp"));//ok
set.add(new String("hsp"));//加入不了.
```

`HashSet`在Java中是基于`HashMap`实现的，它保证集合中的元素唯一性是通过元素的`hashCode()`方法和`equals()`方法来实现的。当你尝试向`HashSet`中添加一个元素时，`HashSet`会首先计算元素的哈希码，使用这个哈希码找到存储位置来判断元素是否已经存在于集合中。如果没有相同的哈希码，元素被认为是唯一的，因此可以添加。如果有相同的哈希码，还会调用`equals()`方法来检查两个元素是否真正相等。只有当哈希码相同且`equals()`方法返回`true`时，元素才被认为是重复的，因此不会被添加。

- 当尝试两次添加字符串`"lucy"`到`HashSet`中时，第二次添加失败。这是因为字符串在Java中是不可变的，且它们的`hashCode()`和`equals()`方法被设计为比较字符串的实际内容。因此，两个包含相同字符序列的字符串对象被视为相等。
- 对于`new Dog("tom")`，除非你在`Dog`类中重写了`hashCode()`和`equals()`方法来基于`Dog`对象的属性（如名字）进行比较，否则每次`new Dog("tom")`都会创建一个具有不同哈希码的新对象。由于默认的`hashCode()`方法是根据对象的内存地址生成哈希码的（这在Java的`Object`类中定义），不同的`Dog`实例将会有不同的哈希码，即使它们的属性相同。同样，默认的`equals()`方法比较的是对象的引用是否相同，而不是它们的内容。因此，除非你重写了这些方法，否则两个属性相同的`Dog`对象会被认为是不同的，从而都能被添加到`HashSet`中。



### Map接口和常用方法

#### Map接口实现类的特点【很实用】

【JDK 8 Map 接口特点】

1）Map与Collection并列存在，用于保存具有映射关系的数据：`Key-Value`

2）Map中的key和value可以是任何引用类型的数据，会封装到HashMap$Node对象中

3）Map中的key不允许重复，原因和HashSet一样

4）Map中的value可以重复

5）Map的key可以为null，value也可以为null，注意key为null，只能有一个，value为null可以多个

6）常用String类作为Map的key

7）key和value之间存在**单向**一对一关系，即通过指定的key总能找到对应的value

#### 常用方法

以下是`Map`接口中一些常用方法的概述：

- **put(K key, V value)**: 将指定的值与此映射中的指定键关联（可选操作）。
- **get(Object key)**: 返回指定键所映射的值，如果此映射不包含该键的映射，则返回`null`。
- **remove(Object key)**: 如果存在一个键的映射，则将其从此映射中移除（可选操作）。
- **containsKey(Object key)**: 如果此映射包含指定键的映射，则返回`true`。
- **containsValue(Object value)**: 如果此映射将一个或多个键映射到指定值，则返回`true`。
- **keySet()**: 返回此映射中包含的键的`Set`视图。
- **values()**: 返回此映射中包含的值的`Collection`视图。
- **entrySet()**: 返回此映射中包含的映射关系的`Set`视图。
- **size()**: 返回此映射中的键值对数量。
- **isEmpty()**: 如果此映射不包含键值对，则返回`true`。
- **clear()**: 从此映射中移除所有映射关系（可选操作）。

```java
import java.util.HashMap;
import java.util.Map;

public class MapExample {
    public static void main(String[] args) {
        // 创建HashMap实例
        Map<String, Integer> map = new HashMap<>();
        
        // 添加键值对到Map
        map.put("Apple", 1);
        map.put("Banana", 2);
        map.put("Cherry", 3);
        
        // 获取并打印特定键的值
        System.out.println("Value for 'Apple': " + map.get("Apple"));
        
        // 检查Map是否包含特定的键或值
        System.out.println("Contains key 'Banana': " + map.containsKey("Banana"));
        System.out.println("Contains value 2: " + map.containsValue(2));
        
        // 移除键值对
        map.remove("Cherry");
        
        // 遍历Map
        for (Map.Entry<String, Integer> entry : map.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
    }
}

```

### Map接口实现类-HashMap

#### HashMap小结

1）Map接口的常用实现类：HashMap、Hashtable和Properties

2）HashMap是Map接口使用频率最高的实现类

3）HashMap是以key-val对的方式存储数据

4）key不能重复，但是值可以重复，允许使用null键和null值

5）如果添加相同的key，则会覆盖原来的key-val，等同于修改（key不会替换，val会替换）

6）与HashSet一样，不保证映射的顺序，因为底层是以hash表的方式来存储的（jdk8的hashmap底层数组+链表+红黑树）

7）hashmap没有实现同步，因此是线程不安全的，方法没有做同步互斥的操作

#### HashMap底层机制

![image-20240227104454937](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202402271044136.png)

![image-20240227104505930](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202402271045097.png)

`HashMap`是Java中基于哈希表的Map接口的非同步实现，提供了所有可选的映射操作，并允许使用null值和null键。`HashMap`类在java.util包下，它存储的内容是键值对(key-value pairs)。下面是`HashMap`的一些常用方法：

#### 基本操作

- **put(K key, V value)**: 将指定的键值对添加到映射中。如果映射之前包含该键的映射关系，则旧值将被替换。
- **get(Object key)**: 返回指定键所映射的值；如果此映射不包含该键的映射关系，则返回null。
- **remove(Object key)**: 从映射中移除指定键的映射关系（如果存在）。
- **containsKey(Object key)**: 如果此映射包含指定键的映射关系，则返回true。
- **containsValue(Object value)**: 如果此映射将一个或多个键映射到指定值，则返回true。
- **size()**: 返回映射中的键值对数量。
- **isEmpty()**: 如果映射不包含键值对，则返回true。

#### 批量操作

- **putAll(Map<? extends K,? extends V> m)**: 将指定映射的所有映射关系复制到此映射中。
- **clear()**: 从此映射中移除所有映射关系。

#### 集合视图

- **keySet()**: 返回此映射中包含的键的Set视图。
- **values()**: 返回此映射中包含的值的Collection视图。
- **entrySet()**: 返回此映射中包含的映射关系的Set视图。

#### 示例代码

下面是使用`HashMap`的一些基本操作的示例：

```java
import java.util.HashMap;

public class HashMapExample {
    public static void main(String[] args) {
        // 创建HashMap实例
        HashMap<String, Integer> map = new HashMap<>();

        // 添加键值对
        map.put("Alice", 10);
        map.put("Bob", 20);
        map.put("Charlie", 30);

        // 访问元素
        System.out.println("Value for Bob: " + map.get("Bob"));

        // 检查存在性
        if (map.containsKey("Alice")) {
            System.out.println("Alice is in the map");
        }

        // 移除元素
        map.remove("Charlie");

        // 遍历键值对
        for (HashMap.Entry<String, Integer> entry : map.entrySet()) {
            System.out.println(entry.getKey() + " => " + entry.getValue());
        }
    }
}

```

### Map接口实现类-HashTable

#### HashTable的基本介绍

1）存放的元素是键值对：即k-v

2）hashtable的键和值都不能为null，否则会抛出`NullPointerException`

3）hashTable使用方法基本上和HashMap一样

4）hashTable是线程安全的，hashMap是线程不安全的

![image-20240227110711662](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202402271107811.png)

```java
Hashtable table = new Hashtable(); //
table.put("john", 100);
table.put(null, 100);
table.put("john", null);
table.put("lucy", 100);
table.put("lic", 100);
table.put("lic", 88);
```

1. `Hashtable table = new Hashtable();` // 正确，创建了一个`Hashtable`实例。
2. `table.put("john", 100);` // 正确，将键值对（"john", 100）添加到哈希表中。
3. `table.put(null, 100);` // 错误，`Hashtable`不允许使用null作为键。
4. `table.put("john", null);` // 错误，`Hashtable`同样不允许使用null作为值。
5. `table.put("lucy", 100);` // 正确，将键值对（"lucy", 100）添加到哈希表中。
6. `table.put("lic", 100);` // 正确，但您的代码中缺少了双引号，应该是`table.put("lic", 100);`。
7. `table.put("lic", 88);` // 正确，这会替换掉键为"lic"的条目的值，新值为88。
8. `System.out.println(table);` // 正确，这将输出`Hashtable`中的所有条目。

### Map接口实现类-Porperties

#### 基本介绍

1）Properties类继承自Hashtable并且实现了Map接口，也是使用一种键值对的形式来保存数据

2）他的使用特点和Hashtable类似

3）Properties还可以用于 从`xxx.properties`文件中，加载数据到Properties类对象，并进行读取和修改

4）说明：工作后 `xxx.properties`文件通常作为配置文件

### 总结：开发中如何选择集合实现类（记住）

在开发中，选择什么集合实现类，主要取决于业务操作特点，然后根据集合实现类特性进行选择，分析如下：

1）先判断存储的类型（一组对象[单列]或一组键值对[双列]）

2）一组对象[单列]：Collection接口
**允许重复**：List

增删多：LinkedList（底层维护一个双向链表）

改查多：ArrayList（底层维护Object类型的可变数组）
**不允许重复**：Set

无序：HashSet（底层是HashMap，维护一个哈希表）

排序：TreeSet

插入和取出顺序一致：LinkedHashSet（维护数组+双向链表）

3）一组键值对[双列]：Map

键无序：HashMap

键排序：TreeMap

键插入和取出顺序一致：LinkedHashMap

读取文件：Properties



## 泛型

### 泛型的介绍

Java中的泛型机制是一种在编译时期定义类、接口和方法代码的方式，它可以让你在创建集合、定义类或者方法的时候指定它们可以操作的数据类型。泛型的好处是它们在编译时期强制执行类型安全性，减少了需要进行的类型转换，并可以检测到在编译期间可能出现的类型兼容性错误。

### 泛型的理解和好处

1）编译时，检查添加元素的类型，提高了安全性

2）减少了类型转换的次数，提高效率

![image-20240227152201089](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202402271522240.png)

3）不再提示编译警告

### 泛型语法

泛型的声明：

```java
// 泛型接口
interface InterfaceName<T>{}
// 泛型类
class ClassName<T>{}
// 泛型方法
public class Util {
    // 以下是泛型方法的声明
    public static <T> void printArray(T[] inputArray) {
        for (T element : inputArray) {
            System.out.printf("%s ", element);
        }
        System.out.println();
    }
}
// 说明T表示type不代表值，表示类型
```

泛型的实例化

```java
要在类名后面指定类型参数的值（类型）如：
List<String> strList = new ArrayList<String>();
Iterator<Customer> iterator = customers.iterator();
```

### 泛型使用的注意事项和细节

1.`interface List<T>{}`,`public class HashSet<E>{}`等等

T，E只能是引用类型

```java
List<Integer> list = new ArrayList<Integer>(); // right
List<int> list = new ArrayList<int>(); // wrong
```

2.在给泛型指定具体类型后，可以传入该类型或者其子类类型

### 自定义泛型类

注意细节：

1）普通成员可以使用泛型（属性，方法）

2）使用泛型的数组，不能初始化

3）静态方法中不能使用类的泛型

4）泛型类的类型，实在创建对象时确定的

5）如果在创建对象时，没有指定类型，默认为Object



## 多线程基础

### 线程基本使用

创建线程的两种方式：

1. 继承Thread类，重写run方法
2. 实现Runnable接口，重写run方法

### 线程应用案例1-继承Thread类

1. **定义一个类继承`Thread`类**。
2. **重写`Thread`类的`run()`方法**，将要在新线程中执行的代码放在这个方法中。
3. **创建继承了`Thread`类的对象**。
4. **调用这个对象的`start()`方法**来启动新线程。

```java
// 1. 继承Thread类
class MyThread extends Thread {
    private int ticket = 10;

    // 2. 重写run方法
    @Override
    public void run() {
        while (ticket > 0) {
            // 模拟售票操作
            System.out.println(Thread.currentThread().getName() + " - 售票：票号 " + ticket);
            ticket--;
            // 模拟延迟
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}

public class ThreadExample {
    public static void main(String[] args) {
        // 3. 创建线程对象
        MyThread t1 = new MyThread();
        MyThread t2 = new MyThread();
        MyThread t3 = new MyThread();

        // 4. 启动线程
        t1.start(); // 启动第一个线程
        t2.start(); // 启动第二个线程
        t3.start(); // 启动第三个线程
    }
}

```

在上面的代码中，我们创建了一个`MyThread`类，它继承自`Thread`类并重写了`run()`方法来实现售票的功能。在`main`方法中，我们实例化了三个`MyThread`对象并分别启动了三个线程。

需要注意的是，每次调用`start()`方法时，Java虚拟机会为每个线程分配新的调用栈。`run()`方法的执行结束并不意味着整个线程的生命周期结束，线程还需要完成一些清理工作才能真正结束。

虽然继承`Thread`类可以实现多线程，但在Java中更推荐实现`Runnable`接口的方式来创建线程，因为这种方式更加灵活，不会因为Java的单继承限制而受到约束，同时它也更适合于多个线程执行相同的任务的场景。

### 线程应用案例2-实现Runnable接口

1. java是单继承，在某些情况下一个类可能已经继承了某个父类，这时再用继承Thread类方法来创建线程显然不可能。
2. java设计者提供了另外一个方式创建线程，通过Runnable接口创建不会受到单继承限制

步骤：

1. **定义一个类实现`Runnable`接口**。
2. **实现`Runnable`接口的`run()`方法**，在这个方法中放入你希望在新线程中执行的代码。
3. **创建`Thread`类的实例**，在构造函数中传入你的`Runnable`实现类的实例。
4. **调用`Thread`实例的`start()`方法**来启动新线程。

```java
// 1. 实现Runnable接口
class MyRunnable implements Runnable {
    private int ticket = 10;

    // 2. 实现run方法
    @Override
    public void run() {
        while (ticket > 0) {
            // 模拟售票操作
            System.out.println(Thread.currentThread().getName() + " - 售票：票号 " + ticket);
            ticket--;
            // 模拟延迟
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}

public class RunnableExample {
    public static void main(String[] args) {
        // 创建Runnable实现类的实例
        MyRunnable myRunnable = new MyRunnable();

        // 3. 创建Thread类的实例，并将Runnable实例传递给Thread的构造器
        Thread t1 = new Thread(myRunnable);
        Thread t2 = new Thread(myRunnable);
        Thread t3 = new Thread(myRunnable);

        // 4. 调用start()方法启动线程
        t1.start(); // 启动第一个线程
        t2.start(); // 启动第二个线程
        t3.start(); // 启动第三个线程
    }
}

```

### 线程终止

基本说明：

1. 当线程完成任务后，会自动退出。
2. 还可以通过**使用变量**来控制run方法退出的方式停止线程，即**通知方式**

推荐的终止线程的方式是使用一个标志位来通知线程。这是一种合作机制，其中运行线程周期性地检查一个标志位来决定是否要终止。这种方法比使用`Thread.stop()`更安全，因为它不会导致锁定的对象突然解锁，从而避免了不一致的状态。

```java
public class SafeStopThread implements Runnable {
    // 标志位
    private volatile boolean exit = false;

    public void run() {
        while (!exit) {
            // 执行任务
            System.out.println("线程正在运行...");
            try {
                Thread.sleep(1000); // 模拟长时间任务
            } catch (InterruptedException e) {
                // 如果线程在sleep状态下停止，会抛出InterruptedException
                // 这里可以捕获这个异常，来进一步处理线程的停止逻辑
                System.out.println("线程被中断");
                // 重新设置中断状态
                Thread.currentThread().interrupt();
            }
        }
        System.out.println("线程安全地停止了");
    }

    // 用于设置标志位的公共方法
    public void stopThread() {
        exit = true;
    }

    public static void main(String[] args) throws InterruptedException {
        SafeStopThread task = new SafeStopThread();
        Thread thread = new Thread(task);
        thread.start();

        // 模拟应用程序运行了一段时间后需要停止线程
        Thread.sleep(3000); // 运行3秒后停止线程
        task.stopThread();
    }
}

```

在这个例子中，我们定义了一个`SafeStopThread`类实现了`Runnable`接口。我们添加了一个`volatile`修饰的布尔变量`exit`作为标志位。在`run`方法中，我们检查这个标志位来决定是否退出循环。有一个`stopThread`方法用来设置`exit`变量为`true`，这个方法可以被外部调用来安全地停止线程。

当`main`方法启动线程后，它会等待3秒钟，然后调用`stopThread`方法来设置`exit`为`true`，通知线程终止。线程检查到`exit`为`true`后，会跳出循环并打印消息表示线程已安全停止。这种方法可以确保线程的资源被适当释放，而不会导致不一致的状态。



### 线程常用方法

在Java中，线程（Thread）类提供了多种方法来管理和控制线程的执行。以下是一些常用的Thread类方法：

1. **start()**: 启动线程，使其成为可运行状态。真正的执行由线程调度器按照某种选择机制进行。
2. **run()**: 新线程创建之后，将执行此方法中定义的代码。
3. **sleep(long millis)**: 静态方法，使当前正在执行的线程停留（暂时停止执行）指定的毫秒数，进入阻塞状态。
4. **join()**: 等待该线程终止。使用此方法会使当前线程进入等待状态，直到调用join方法的线程结束运行。
5. **interrupt()**: 中断线程。这实际上设置了线程的中断状态标志，线程被中断时，并不会立即停止执行，而是给线程一个机会来响应中断。
6. **isInterrupted()**: 测试线程是否被中断（线程的中断状态）。
7. **static interrupted()**: 测试当前线程是否被中断（检查中断状态并清除中断状态标志）。
8. **yield()**: 静态方法，暗示线程调度器当前线程愿意放弃其当前的CPU使用（但这不是强制的，调度器可能会忽略这个暗示）。
9. **setDaemon(boolean on)**: 将该线程标记为守护线程或用户线程。当运行的线程都是守护线程时，Java虚拟机将退出。
10. **isDaemon()**: 检查线程是否是守护线程。
11. **setName(String name)**: 更改线程名称，使之更易于理解。
12. **getName()**: 返回线程的名称。
13. **getPriority()**: 返回线程的优先级。
14. **setPriority(int newPriority)**: 更改线程的优先级。
15. **currentThread()**: 静态方法，返回当前正在执行的线程对象的引用。
16. **getId()**: 返回线程的ID。
17. **getState()**: 返回线程的状态（NEW, RUNNABLE, BLOCKED, WAITING, TIMED_WAITING, TERMINATED）。
18. **setUncaughtExceptionHandler(Thread.UncaughtExceptionHandler eh)**: 设置当线程由于未捕获的异常而突然终止，并且没有其他处理程序时要调用的默认处理程序。
19. **getUncaughtExceptionHandler()**: 返回设置的异常处理器。

线程的生命周期管理主要依赖于这些方法，使得开发者能够编写出并发运行的程序。这些方法结合`Runnable`接口或直接继承`Thread`类，是实现多线程应用程序的基础。

### 线程的生命周期及线程的状态转换


Java线程的生命周期包含几个状态，以及这些状态之间的转换。线程状态在`java.lang.Thread.State`枚举中定义，其主要状态包括：

1. **NEW**: 新创建的线程，尚未开始执行。
2. **RUNNABLE**: 在Java虚拟机中执行的线程，可能正在运行也可能正在等待CPU分配时间片。
3. **BLOCKED**: 被阻塞等待监视器锁的线程，处于同步块或方法中被阻塞。
4. **WAITING**: 无限期等待另一个线程执行特定操作的线程。
5. **TIMED_WAITING**: 等待另一个线程执行动作直到达到指定等待时间的线程。
6. **TERMINATED**: 已退出的线程。

线程状态之间的转换路径如下：

- **创建后启动**: 当线程被创建，即实例化后调用`start()`方法，线程状态从**NEW**变为**RUNNABLE**。
- **执行中**: 在**RUNNABLE**状态的线程可能在执行也可能等待CPU调度。
- **等待状态**: 当线程执行了`Object.wait()`, `Thread.join()`, `LockSupport.park()`等方法时，它会进入**WAITING**状态。如果是调用的`wait()`, `join()`或`sleep()`等方法中带有时间限制的版本，则线程进入**TIMED_WAITING**状态。
- **阻塞**: 如果线程尝试获取一个内部的对象锁（不是`java.util.concurrent`包中的锁），而该锁被其他线程持有，则该线程进入**BLOCKED**状态。
- **唤醒/通知或超时**: 处于**WAITING**或**TIMED_WAITING**状态的线程，如果其他线程调用了`notify()`, `notifyAll()`或者指定的等待时间已到，它会重新变为**RUNNABLE**状态。
- **中断**: 处于阻塞，等待，或定时等待状态的线程，如果被其他线程调用`interrupt()`方法，它会抛出`InterruptedException`，并返回到**RUNNABLE**状态。
- **结束**: 线程的`run()`方法执行完成后，它将进入**TERMINATED**状态。

线程一旦终结，就不能再次启动。线程状态的这些转换是由操作系统的线程调度程序、Java虚拟机以及应用程序代码的交互作用来控制的。了解线程的生命周期对于编写多线程并发程序至关重要。

### 线程的同步

1. 在多线程编程中，一些敏感数据不允许被多个线程同时访问，此时就使用同步访问计数，在保证数据在任何同一时刻，最多有一个线程访问，以保证数据的完整性。
2. 也可以这样理解：线程同步，即当有一个线程在对内存进行操作时，其他线程都不可对这个内存地址进行操作，直到该线程完成操作，其他线程才能对该内存地址进行操作。

### 同步具体方法-Synchronized

线程同步的原理基于锁的概念。在Java中，每个对象都有一个内置锁（也称为监视器锁）。当某个线程想要执行同步代码块时，它需要先获取对应对象的锁。如果锁被其他线程持有，则当前线程会被阻塞，直到锁被释放。当锁被释放后，等待的线程之一会获取该锁，然后继续执行。这样，就可以确保在任何时刻，只有一个线程可以访问被同步的资源。

Java提供了几种线程同步的方法：

1. **同步方法**：通过在方法声明中添加`synchronized`关键字来定义。这意味着该方法在执行时会锁定调用它的对象（如果是静态方法，则锁定类的Class对象）。
2. **同步代码块**：通过`synchronized`关键字和指定的锁对象来创建。同步代码块提供了更灵活的方式来控制代码段的同步，可以减少不必要的同步开销。
3. **锁对象（Locks）**：`java.util.concurrent.locks`包提供了一系列的锁实现，如`ReentrantLock`。这些锁提供了比synchronized关键字更高级的锁定功能，包括尝试非阻塞获取锁、可中断的锁获取等。

```java
public class SynchronizedExample {
    private int count = 0;

    // 同步方法
    public synchronized void increment() {
        count++;
    }

    // 同步代码块
    public void incrementBlock() {
        synchronized (this) {
            count++;
        }
    }
}

```

#### 原理小结

- 当线程进入同步方法或同步代码块时，它会自动获取锁。
- 当线程退出同步方法或同步代码块时，它会自动释放锁。
- 如果另一个线程尝试进入已被锁定的方法或代码块，它将被阻塞直到锁被释放。

通过这种机制，Java的线程同步确保了并发执行的线程在访问共享资源时的安全性和一致性。



### 互斥锁

基本介绍

1. Java中，引入了对象互斥锁的概念，来保证共享数据操作的完整性。
2. 每个对象都对应于一个可称为“互斥锁”的标记，这个标记用来保证在任一时刻，只能有一个线程访问该对象。
3. 关键字`synchronized`来与对象的互斥锁联系，当某个对象用`synchronized`修饰时，表明该对象在任一时刻只能有一个线程访问。
4. 同步的局限性：导致程序的执行效率要降低。
5. 同步方法（非静态）的锁可以是`this`，也可以是其他对象（要求是同一个对象）
6. 同步方法（静态）的锁为当前类本身。

#### 注意事项：

1. 同步方法如果没有使用static修饰：默认锁对象为this
1. 如果方法使用static修饰，默认锁对象：`当前类.class`

```java
/*
    下面是一个使用Java中ReentrantLock类（一种实现了Lock接口的互斥锁）解决售票问题的简单示例。这个示例创建了一个售票窗口，多个线程（模拟多个售票员）竞争售票。

首先，定义一个TicketSeller类来模拟售票的过程，其中使用ReentrantLock来确保在售票时，每次只有一个线程能够执行售票操作。
*/
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

class TicketSeller implements Runnable {
    private int ticketsAvailable = 10; // 假设有10张票
    private final Lock lock = new ReentrantLock(); // 创建一个锁对象

    @Override
    public void run() {
        while (true) {
            lock.lock(); // 获取锁
            try {
                if (ticketsAvailable > 0) {
                    System.out.println(Thread.currentThread().getName() + " 正在售卖第 " + ticketsAvailable + " 张票");
                    ticketsAvailable--; // 票数减一
                    Thread.sleep(100); // 模拟售票时间
                } else {
                    System.out.println("票已售罄");
                    break;
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                lock.unlock(); // 释放锁
            }
        }
    }
}

public class TicketSelling {
    public static void main(String[] args) {
        TicketSeller seller = new TicketSeller(); // 创建售票对象
        Thread t1 = new Thread(seller, "售票员1");
        Thread t2 = new Thread(seller, "售票员2");
        Thread t3 = new Thread(seller, "售票员3");

        t1.start();
        t2.start();
        t3.start();
    }
}

```





### 线程的死锁

多个线程都占用了对方的锁资源，但不肯相让，导致了死锁，在编程中是一定要避免发生的



### 线程的释放锁

1. 当前线程的同步方法、同步代码块执行结束
2. 当前线程在同步代码块、同步方法中遇到break，return
3. 当前线程在同步代码块、同步方法中出现了未处理的Error或Exception，导致异常结束
4. 当前线程在同步代码块、同步方法中执行了线程对象的wait()方法，当前线程暂停，并释放锁

以下操作不会释放锁

1. 线程执行同步代码块或同步方法时，程序调用`Thread.sleep()``Thread.yield()`方法暂停当前现成的执行、不会释放锁
2. 线程执行同步代码块时，其他线程调用了该线程的`suspend()`方法将线程挂起，该线程不会释放锁

 
