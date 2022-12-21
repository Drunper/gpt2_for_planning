(define (problem logistics-test)
(:domain logistics)
(:objects
	apn2 apn3 - airplane
	cit3 cit6 - city
	apt6 apt5 - airport
	tru4 tru5 - truck
	obj00 obj12 obj33 obj11 obj99 obj66 - package
	pos22 pos11 pos55 pos23 pos13 pos21 - location
)
(:init
	(at apn2 apt6)
	(at apn3 apt5)
	(at obj00 pos21)
	(at obj12 pos22)
	(at obj33 pos11)
	(at obj11 pos55)
	(at obj99 pos23)
	(at obj66 pos13)
	(in-city apt6 cit3)
	(in-city apt5 cit6)
	(in-city pos22 cit3)
	(in-city pos11 cit6)
	(in-city pos55 cit6)
	(in-city pos23 cit3)
	(in-city pos13 cit3)
	(in-city pos21 cit3)
	(at tru5 pos23)
	(at tru4 pos11)
)
(:goal
	(and
		(at obj12 pos22)
		(at obj11 pos21)
))
)
